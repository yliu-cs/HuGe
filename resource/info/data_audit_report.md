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

### proprietary_real_cola_merged_cleaned_0530

- root: `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/proprietary/real/cola_merged_cleaned_0530`
- root_exists: `True`
- root_is_symlink: `False`
- token_files: `578`
- proprio_files: `578`
- first_token_shape: `440x66`
- first_proprio_shape: `440x34`
- motion_token_finite_sample: `True`
- hand_stats: `{"frames": 10083, "left_closed_ratio": 0.0, "right_closed_ratio": 0.49885946642864226, "invalid_episode_count": 0}`
- errors: `[]`

## Proprio Stats

- `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json`: `{"path": "/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json", "exists": true, "roundtrip_max_abs_error": 3.0616030244345893e-07, "proprio_dim": 34}`
- `/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json`: `{"path": "/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/liuyang/Kappa/resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json", "exists": true, "roundtrip_max_abs_error": 1.3514174668216583e-07, "proprio_dim": 34}`
