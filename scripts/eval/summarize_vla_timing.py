"""Summarize client-local timing JSONL; does not connect to any running service."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        index = int(position)
        return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index)

    return {"count": len(values), "p50": percentile(0.5), "p95": percentile(0.95),
            "p99": percentile(0.99), "max": values[-1]}


def summarize_hand_compat(metadata: dict[str, Any], requests: dict[int, dict[str, Any]],
                          trace_complete: bool) -> dict[str, Any]:
    mode = metadata.get("hand_state_compat", {}).get("mode", "off")
    enabled = mode != "off"
    result: dict[str, Any] = {"mode": mode, "enabled": enabled, "valid_records": 0,
                              "missing_request_ids": [], "invalid_request_ids": [],
                              "audit_complete": None,
                              "note": "Old-rule compatibility values, not actual grasp success."}
    sides = {side: {"changed": 0, "before": {"0": 0, "1": 0}, "after": {"0": 0, "1": 0}}
             for side in ("left", "right")}
    result["sides"] = sides
    if not enabled:
        return result
    for rid, request in requests.items():
        audit = request.get("hand_state_compat")
        if audit is None:
            result["missing_request_ids"].append(rid)
            continue
        try:
            valid = audit["mode"] == mode and all(
                len(audit[key]) == 2 and all(value in (0, 1) for value in audit[key])
                for key in ("before", "after")) and all(
                len(audit[key]) == 7 and all(math.isfinite(value) for value in audit[key])
                for key in ("left_hand_q", "right_hand_q"))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            result["invalid_request_ids"].append(rid)
            continue
        result["valid_records"] += 1
        for i, side in enumerate(("left", "right")):
            sides[side]["changed"] += int(audit["before"][i] != audit["after"][i])
            for key in ("before", "after"):
                sides[side][key][str(int(audit[key][i]))] += 1
    result["audit_complete"] = trace_complete and not result["missing_request_ids"] and not result["invalid_request_ids"]
    return result


def summarize(directory: Path, server_log_dir: Path | None = None) -> dict[str, Any]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("clock") != "CLOCK_MONOTONIC":
        raise ValueError("Unsupported timing schema/clock")
    records = []
    issues = []
    with (directory / "events.jsonl").open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                issues.append(f"Invalid or truncated JSON at line {number}")
    footer = records[-1] if records and records[-1].get("event") == "trace_end" else None
    if footer is None or not footer.get("complete"):
        issues.append("Trace did not finish cleanly or dropped events")
    counts = Counter(row["event"] for row in records)
    requests: dict[int, dict[str, Any]] = {}
    errors = []
    publications = []
    boundaries = []
    pause_changes = [(-1, True)]
    paused = True
    for row in sorted(records, key=lambda value: value["mono_ns"]):
        kind = row["event"]
        if kind == "keyboard" and row["key"] in ("p", "i", "k"):
            boundaries.append(row["mono_ns"])
            if row["key"] == "p":
                paused = not paused
                pause_changes.append((row["mono_ns"], paused))
        if "error" in row or kind in ("result_unavailable", "client_exit"):
            errors.append(row)
        rid = row.get("request_id")
        if rid is not None and kind != "action_send":
            request = requests.setdefault(rid, {"request_id": rid})
            request.update({key: value for key, value in row.items() if key not in ("event", "mono_ns")})
            request.setdefault("events", []).append(kind)
        if kind == "action_send":
            if row.get("source") != "model" or row.get("action_send_done_ns") is None:
                boundaries.append(row["action_send_begin_ns"])
            else:
                publications.append(row)
            if row.get("source") == "unattributed":
                issues.append(f"Unattributed publication {row['publish_seq']}")
    publications.sort(key=lambda row: row["action_send_begin_ns"])
    boundaries.sort()
    pause_times = [value[0] for value in pause_changes]
    metrics: dict[str, list[float]] = {key: [] for key in (
        "rpc_roundtrip_ms", "decode_and_convert_ms", "first_publish_wait_ms",
        "request_to_first_publish_ms", "action_send_call_ms", "publish_interval_ms",
        "camera_local_wait_ms",
    )}
    first_publish = publications[0]["action_send_begin_ns"] if publications else None
    phases: dict[str, list[float]] = {"startup": [], "client_paused": [], "publishing": []}

    def delta(target: str, end: int | None, begin: int | None) -> float | None:
        if end is None or begin is None:
            return None
        if end < begin:
            issues.append(f"Negative {target}: check event completeness and clock")
            return None
        value = (end - begin) / 1e6
        metrics[target].append(value)
        return value

    by_request: dict[int, list[dict[str, Any]]] = {}
    repeats = []
    repeat_run = None
    previous = None
    seen_sequences = set()
    for row in publications:
        rid = row["request_id"]
        if row["publish_seq"] in seen_sequences:
            issues.append(f"Duplicate publish_seq {row['publish_seq']}")
        seen_sequences.add(row["publish_seq"])
        if rid not in requests:
            issues.append(f"Publication references missing request {rid}")
        by_request.setdefault(rid, []).append(row)
        begin = row["action_send_begin_ns"]
        delta("action_send_call_ms", row["action_send_done_ns"], begin)
        segment = bisect_right(boundaries, begin)
        if previous is not None and previous[1] == segment:
            prior = previous[0]
            delta("publish_interval_ms", begin, prior["action_send_begin_ns"])
            last_index = requests.get(rid, {}).get("horizon", 0) - 1
            if rid == prior["request_id"] and row["chunk_index"] == prior["chunk_index"] == last_index:
                if repeat_run is None:
                    repeat_run = {"request_id": rid, "repeated_sends": 0,
                                  "begin_ns": prior["action_send_begin_ns"], "end_ns": begin}
                repeat_run["repeated_sends"] += 1
                repeat_run["end_ns"] = begin
            elif repeat_run is not None:
                repeats.append(repeat_run)
                repeat_run = None
        elif repeat_run is not None:
            repeats.append(repeat_run)
            repeat_run = None
        previous = (row, segment)
    if repeat_run is not None:
        repeats.append(repeat_run)
    for repeat in repeats:
        repeat["duration_ms"] = (repeat.pop("end_ns") - repeat.pop("begin_ns")) / 1e6

    camera_reuse = 0
    camera_comparisons = 0
    previous_camera = None
    never_published = []
    unfinished = []
    for rid, row in sorted(requests.items()):
        t0 = row.get("request_send_begin_ns")
        t1 = row.get("response_recv_done_ns")
        ready = row.get("result_ready_ns")
        rtt = delta("rpc_roundtrip_ms", t1, t0)
        if t0 is not None:
            phase = "startup" if first_publish is None or t0 < first_publish else (
                "client_paused" if pause_changes[bisect_right(pause_times, t0) - 1][1] else "publishing")
            row["phase"] = phase
            if rtt is not None:
                phases[phase].append(rtt)
        delta("decode_and_convert_ms", ready, t1)
        delta("camera_local_wait_ms", t0, row.get("camera_read_done_ns"))
        image_seq = row.get("camera_message_seq")
        if image_seq is not None:
            if previous_camera is not None:
                camera_comparisons += 1
                camera_reuse += image_seq == previous_camera
            previous_camera = image_seq
        sent = by_request.get(rid, [])
        row["published_steps"] = len(sent)
        if sent:
            first = sent[0]
            row["first_chunk_index"] = first["chunk_index"]
            row["first_publish_seq"] = first["publish_seq"]
            delta("first_publish_wait_ms", first["action_send_begin_ns"], ready)
            delta("request_to_first_publish_ms", first["action_send_begin_ns"], t0)
        elif ready is not None:
            never_published.append(rid)
        if ready is None and not any(kind in row["events"] for kind in (
            "request_error", "result_error", "result_unavailable",
        )):
            unfinished.append(rid)
    if unfinished:
        issues.append("Some requests were still unfinished at end of trace")
    if not counts["client_end"]:
        issues.append("Missing client_end event")

    complete = not issues
    report = {
        "trace_session_id": metadata["trace_session_id"], "complete": complete,
        "issues": issues, "event_counts": dict(counts),
        "dropped_events": footer.get("dropped_events") if footer else None,
        "metrics": {key: distribution(value) for key, value in metrics.items()},
        "rpc_roundtrip_ms_by_phase": {key: distribution(value) for key, value in phases.items()},
        "camera_reused_pairs": camera_reuse, "camera_compared_pairs": camera_comparisons,
        "ready_without_publication": never_published if complete else None,
        "unpublished_candidates_in_incomplete_trace": never_published if not complete else [],
        "unfinished_requests": unfinished, "tail_repeats": repeats,
        "requests": list(requests.values()), "errors": errors,
        "hand_state_compat": summarize_hand_compat(metadata, requests, complete),
        "limits": [
            "PUB send completion is not WBC reception or execution acknowledgement.",
            "RPC time includes transport, server waiting/processing and logging; not GPU-only inference.",
            "Camera local wait starts after userspace recv, not exposure or NIC arrival.",
            "Publishing phase reflects client keyboard state, not confirmed WBC mode.",
        ],
    }
    if server_log_dir is not None:
        import msgpack
        import msgpack_numpy

        matches: dict[int, list[str]] = {}
        for path in sorted((server_log_dir / "calls").glob("*.msgpack")):
            record = msgpack.unpackb(path.read_bytes(), raw=False, object_hook=msgpack_numpy.decode)
            tag = (record.get("request", {}).get("data", {}).get("options") or {}).get("huge_timing", {})
            if tag.get("trace_session_id") == metadata["trace_session_id"]:
                matches.setdefault(tag["request_id"], []).append(str(path))
        report["server_calls"] = matches
        report["requests_without_server_match"] = sorted(set(requests) - set(matches))
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# VLA 客户端计时汇总", "",
             f"会话：`{report['trace_session_id']}`。日志完整：`{report['complete']}`。", "",
             "只测量客户端本地时序；PUB 返回不等于 WBC 接收或机器人执行。", "",
             "| 指标（ms） | 样本数 | p50 | p95 | p99 | 最大值 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, values in report["metrics"].items():
        cells = [str(values["count"])] + ["—" if values[q] is None else f"{values[q]:.3f}" for q in ("p50", "p95", "p99", "max")]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", f"请求数：{len(report['requests'])}；模型动作发布数：{sum(r['published_steps'] for r in report['requests'])}。",
              f"缓存图像复用：{report['camera_reused_pairs']}/{report['camera_compared_pairs']} 对相邻请求。",
              f"已就绪但未观察到发布的请求：{report['ready_without_publication']}（日志不完整时不下结论）。",
              f"末尾仍未结束的请求：{report['unfinished_requests']}。",
              f"连续尾帧重复：{sum(r['repeated_sends'] for r in report['tail_repeats'])} 次。", "",
              "完整性问题：" + ("；".join(report["issues"]) or "无"), "",
              "## 请求往返时间按阶段统计", "",
              "startup 为首次模型发布之前；其后按客户端 p 键状态区分暂停和发布，不能证明 WBC 模式。", ""]
    for phase, values in report["rpc_roundtrip_ms_by_phase"].items():
        lines.append(f"- {phase}: {json.dumps(values)}")
    hand = report["hand_state_compat"]
    lines += ["", "## 手状态兼容", ""]
    if not hand["enabled"]:
        lines.append("未启用手状态兼容（包括没有兼容元数据的旧 trace）。")
    else:
        lines += [f"模式：`{hand['mode']}`；兼容审计完整：`{hand['audit_complete']}`。",
                  f"有效记录：{hand['valid_records']}；缺少审计的请求：{len(hand['missing_request_ids'])}；无效审计：{len(hand['invalid_request_ids'])}。",
                  "after 为训练旧规则兼容值，不代表真实握拳或抓握成功。", "",
                  "| 手 | 修改次数 | before=0 | before=1 | after=0 | after=1 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for side, label in (("left", "左"), ("right", "右")):
            values = hand["sides"][side]
            lines.append(f"| {label} | {values['changed']} | {values['before']['0']} | {values['before']['1']} | {values['after']['0']} | {values['after']['1']} |")
    lines += ["", "详细请求索引、异常和可选服务端关联见 summary.json。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--server-log-dir", type=Path)
    args = parser.parse_args()
    directory = args.trace_dir.resolve()
    groot = Path(__file__).resolve().parents[2] / "GR00T-WholeBodyControl"
    if directory == groot or groot in directory.parents:
        parser.error("--trace-dir must be outside GR00T-WholeBodyControl")
    report = summarize(directory, args.server_log_dir)
    (directory / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (directory / "summary.md").write_text(markdown(report))
    print(f"Summary: {directory / 'summary.md'} (complete={report['complete']})")


if __name__ == "__main__":
    main()
