# Kappa Data Audit Report

repo_root: `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa`
sonic_encoder_exists: `True`
qwen_processor_path_exists: `True`

## Sources

### egodex

- root: `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/EgoDex`
- root_exists: `True`
- root_is_symlink: `True`
- token_files: `>=1000`
- proprio_files: `0`
- first_token_shape: `2743x64`
- first_proprio_shape: `missing`
- motion_token_finite_sample: `True`
- hand_stats: `{"frames": 0, "left_closed_ratio": null, "right_closed_ratio": null, "invalid_episode_count": 0}`
- errors: `[]`

### humanoid_everyday

- root: `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/humanoid-everyday`
- root_exists: `True`
- root_is_symlink: `True`
- token_files: `4064`
- proprio_files: `4064`
- first_token_shape: `775x66`
- first_proprio_shape: `775x34`
- motion_token_finite_sample: `True`
- hand_stats: `{"frames": 6519, "left_closed_ratio": 0.38410799202331647, "right_closed_ratio": 0.5157232704402516, "invalid_episode_count": 0}`
- errors: `[]`

### g1_wbt

- root: `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/G1_WBT`
- root_exists: `True`
- root_is_symlink: `True`
- token_files: `3656`
- proprio_files: `3656`
- first_token_shape: `423x66`
- first_proprio_shape: `423x34`
- motion_token_finite_sample: `True`
- hand_stats: `{"frames": 3545, "left_closed_ratio": 1.0, "right_closed_ratio": 0.9997179125528914, "invalid_episode_count": 0}`
- errors: `[]`

### proprietary_real_2026_05_28

- root: `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/proprietary/real/2026-05-28`
- root_exists: `True`
- root_is_symlink: `False`
- token_files: `388`
- proprio_files: `388`
- first_token_shape: `1252x66`
- first_proprio_shape: `1252x34`
- motion_token_finite_sample: `True`
- hand_stats: `{"frames": 9785, "left_closed_ratio": 0.0, "right_closed_ratio": 0.4347470618293306, "invalid_episode_count": 0}`
- errors: `[]`

## Proprio Stats

- `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json`: `{"path": "/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json", "exists": true, "roundtrip_max_abs_error": 3.0616030244345893e-07, "proprio_dim": 34}`
- `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/proprietary/real/2026-05-28/meta/proprio_stats_union_sonic.json`: `{"path": "/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/proprietary/real/2026-05-28/meta/proprio_stats_union_sonic.json", "exists": true, "roundtrip_max_abs_error": 1.0105991776754308e-07, "proprio_dim": 34}`
