# Lab1 Report

## 运行环境

测试平台：Windows 台式机（本地实验环境）

 - CPU: Intel(R) Core(TM) i7-14700K, 20C20T @ 3.4GHz
 - Memory: 32GB
 - COLMAP 版本：4.1.0.dev0 (Commit 5b76f53, with CUDA)
 - GPU: NVIDIA GeForce RTX 4070 Ti SUPER（使用 GPU 加速）

## 题目一：静态场景 SfM

题目一的流程由 `ffmpeg + COLMAP` 完成。实现上，先用 `ffmpeg` 按固定 `fps` 从视频均匀抽帧，再依次调用 `COLMAP` 的 `feature_extractor`、`sequential_matcher`、`mapper` 和 `model_converter` 得到稀疏重建结果，最后根据 `images.txt` 反算相机中心并绘制轨迹图。实验中使用了 `single_camera=1` 和 `PINHOLE` 相机模型，抽帧策略采用等时间间隔采样。

为了比较抽帧策略对结果的影响，三段视频都测试了 `4 / 8 / 16 / 30 fps` 四组设置。对齐叠加图则使用 `uv run lab1 task1 merge` 输出。

### 拼图展示

`S1-1` 在四组帧率下的轨迹差异很明显。`4 fps` 和 `30 fps` 的注册率都偏低，`16 fps` 的轨迹最完整，回环和高度变化也最清楚，因此这一段最适合中等偏高的抽帧率。

![S1-1 fps collage](report_assets/task1/S1-1_fps_grid.png)

`S1-2` 在四组帧率下都得到了稳定的环绕式轨迹，注册率始终为 **100%**。这一段说明场景本身约束充分，抽帧率更多影响的是轨迹稠密度和运行时间，而不是能否成功重建。**观看视频，`S1-2` 几乎始终对准同一批物体，并且这批物体中，不同的物体之间纹理差异较大，且在视角上分布均匀**。

![S1-2 fps collage](report_assets/task1/S1-2_fps_grid.png)

`S1-3` 对抽帧率最敏感。`8 fps` 和 `30 fps` 可以得到完整轨迹，`4 fps` 和 `16 fps` 则明显退化。这一段更接近单方向扫描，因此相邻视角跨度是否合适会直接影响匹配连续性。

![S1-3 fps collage](report_assets/task1/S1-3_fps_grid.png)

### 三段视频在 30 fps 下的轨迹与相机朝向

下图展示 `S1-1`、`S1-2` 和 `S1-3` 在 `30 fps` 设置下的相机轨迹图。蓝色折线表示相机中心轨迹，红色箭头表示沿轨迹均匀采样得到的相机朝向。

![Task1 30fps direction triptych](report_assets/task1/task1_30fps_direction_triptych.png)

> 神奇的是， `S1-2` 的重建坐标系的上下方向反了，摄像机全程“朝上”。

### 合并图展示

`S1-1` 的合并图显示，`8 fps`、`16 fps` 和 `30 fps` 在主干部分大体一致。

![S1-1 merged overlay](report_assets/task1/merged/S1-1_trajectory_overlay.png)

`S1-2` 的合并图几乎完全重合。四组抽帧率在对齐后都沿着同一条主轨迹分布，这和它在定量表中始终保持 **100%** 注册率的结果一致，说明该场景最稳定。

![S1-2 merged overlay](report_assets/task1/merged/S1-2_trajectory_overlay.png)

`S1-3` 的合并图差异最大。`8 fps` 与 `30 fps` 的主轨迹虽然仍有重合区域，但整体分叉明显。

![S1-3 merged overlay](report_assets/task1/merged/S1-3_trajectory_overlay.png)

### 定量比较

注册率和 SfM 时间随抽帧率变化的统计结果如下。可以直接看出，**运行时间基本随 fps 增长而上升，但注册率并不单调变好**。这说明抽帧率不是越高越好，合理的采样密度比盲目增加帧数更重要。

![Task1 fps sweep](report_assets/task1/task1_fps_sweep.png)

| 视频 | fps | 抽帧数 | 注册帧数 | 注册率 | SfM时间/s |
|---|---:|---:|---:|---:|---:|
| S1-1 | 4  | 182  | 63  | 0.346 | 18.94 |
| S1-1 | 8  | 363  | 252 | 0.694 | 76.18 |
| S1-1 | 16 | 726  | 681 | 0.938 | 280.49 |
| S1-1 | 30 | 1362 | 482 | 0.354 | 441.65 |
| S1-2 | 4  | 276  | 276 | 1.000 | 71.94 |
| S1-2 | 8  | 552  | 552 | 1.000 | 135.47 |
| S1-2 | 16 | 1104 | 1104 | 1.000 | 668.25 |
| S1-2 | 30 | 2070 | 2070 | 1.000 | 878.84 |
| S1-3 | 4  | 100  | 44  | 0.440 | 68.21 |
| S1-3 | 8  | 200  | 200 | 1.000 | 1182.61 |
| S1-3 | 16 | 400  | 152 | 0.380 | 522.49 |
| S1-3 | 30 | 750  | 750 | 1.000 | 5936.41 |

## Task 2: Subsequence Pose Analysis

Task 2 still uses the full reconstruction of `S1-2` as reference. We compare two pose extraction modes: **Method A** slices poses directly from the full `images.txt`, while **Method B** reconstructs only the subsequence images independently. We align them with Sim(3) on common registered frames, then compute ATE and trajectory-shape metrics. New runs are added for `fps=8` and `fps=16`, and compared with existing `fps=30`.

The three subsequences represent a medium return segment, a stable scan segment, and a long return segment. This design covers local loop-like motion, one-way scanning, and long-path accumulation, so it directly exposes how sampling density affects independent SfM stability.

### Quantitative Results (new fps8/fps16)

| fps | subsequence | subset_frames | common_registered | ATE | scale | endpoint_distance | path_length | endpoint_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8  | `seq01_return_mid_000211-000930` | 342 | 342 | 0.0243 | 0.7662 | 5.0496 | 19.0368 | 0.2653 |
| 8  | `seq02_scan_stable_000271-000510` | 240 | 240 | 0.0156 | 0.7455 | 4.0574 | 14.1398 | 0.2869 |
| 8  | `seq03_return_long_000031-000930` | 522 | 522 | 0.0208 | 0.9884 | 1.6498 | 30.2798 | 0.0545 |
| 16 | `seq01_return_mid_000211-000930` | 720 | 720 | 0.3171 | 9.8198 | 89.1655 | 220.4054 | 0.4046 |
| 16 | `seq02_scan_stable_000271-000510` | 240 | 240 | 0.1689 | 3.8903 | 43.9332 | 68.7391 | 0.6391 |
| 16 | `seq03_return_long_000031-000930` | 900 | 900 | 0.5309 | 10.8194 | 50.4905 | 290.0049 | 0.1741 |
| 30 | `seq01_return_mid_000211-000930` | 720 | 720 | 0.0723 | 2.3574 | 18.3934 | 47.2025 | 0.3897 |
| 30 | `seq02_scan_stable_000271-000510` | 240 | 240 | 0.0107 | 1.2277 | 15.1095 | 17.3702 | 0.8698 |
| 30 | `seq03_return_long_000031-000930` | 900 | 900 | 0.0668 | 3.0209 | 26.0765 | 63.3635 | 0.4115 |

Across ATE, scale, and endpoint ratio, `fps=8` is the most stable setting overall. It gives the lowest error on `seq01` and `seq03`, and a much smaller endpoint ratio on the long-return segment, indicating stronger shape consistency after alignment. `fps=30` is best only on `seq02_scan_stable`, while `fps=16` degrades significantly on all three segments.

This shows that independent subsequence SfM does not follow a simple "denser is better" rule. Excessive frame redundancy can amplify scale instability and drift instead of strengthening constraints. For this dataset, `fps=8` is the best-balanced sampling density.

## Task 3: Dynamic-Scene SfM

Task 3 compares `raw` reconstruction against masking strategies. The primary criteria are registration ratio, reprojection error, reliable-point ratio, trajectory step behavior, and jump ratio, rather than visual impression of sparse point clouds.

### S2-1 (fps30)

| method | registration_ratio | reliable_ratio | reproj_median | reproj_p90 | track_median | jump_ratio |
|---|---:|---:|---:|---:|---:|---:|
| `raw` | 0.1400 | 0.6319 | 0.7495 | 1.7014 | 7  | 0.0723 |
| `mask_default` | 0.2167 | 0.6189 | 0.5101 | 1.0582 | 54 | 0.0388 |
| `mask_motion` | 0.5683 | 0.6391 | 0.5137 | 1.5263 | 20 | 0.0235 |
| `mask_yolo` | 0.2083 | 0.6601 | 0.7080 | 1.5844 | 10 | 0.0000 |

For `S2-1`, `mask_motion` is the best tradeoff: highest registration ratio with low jump ratio. `mask_yolo` gives the smoothest trajectory (zero jumps) but much lower registration, which behaves like high-precision/low-recall filtering. `raw` has the lowest registration and higher jumps, showing stronger dynamic contamination.

### S2-2 (fps30)

| method | registration_ratio | reliable_ratio | reproj_median | reproj_p90 | track_median | jump_ratio |
|---|---:|---:|---:|---:|---:|---:|
| `raw` | 0.3438 | 0.7094 | 0.6118 | 1.6038 | 6 | 0.0145 |
| `mask_default` | 0.0430 | 0.5860 | 0.5739 | 1.5191 | 7 | 0.0000 |

For `S2-2`, only `raw` and `mask_default` are stably available in the latest outputs. `mask_default` has seemingly moderate error values but only `4.3%` registration, which is a clear reconstruction failure. `raw` is still the only practically usable result.

### Conclusion

Task 2 and Task 3 now support the same core conclusion: **constraints must be balanced; stronger sampling or stronger masking is not automatically better**. In Task 2, `fps=8` is more robust than `fps=16/30`. In Task 3, medium-strength adaptive masking (`mask_motion`) outperforms both weak filtering (`raw`) and overly aggressive masking (`mask_default/yolo`) on `S2-1`.

## Task 4: Pose-Quality Evaluation (annotations 01-10)

Task 4 uses exactly four metric categories, one metric per category:

- smoothness of adjacent poses: `smooth_jump_ratio`
- epipolar consistency of matched points: `epi_dist_px` (symmetric point-to-epiline distance, pixel)
- triangulation reprojection error: `reproj_err_px` (pixel)
- multi-frame composition consistency: `compose_rot_err_deg` (rotation residual of `R_ik` vs `R_jk R_ij`, degree)

Implementation notes:

- feature matching is done from each case `video.mp4` (ORB + BFMatcher) on sampled frames;
- relative pose for pairwise geometry uses Essential matrix (`findEssentialMat + recoverPose`);
- each case summarizes metrics by robust median over valid pairs/triples.

The final quality score mixes these four metrics:

`penalty = 0.25 * (6*smooth_jump_ratio) + 0.25 * log(1 + epi_dist_px/1.5) + 0.25 * log(1 + reproj_err_px/1.5) + 0.25 * log(1 + compose_rot_err_deg/2.0)`

`quality_score = 100 * exp(-penalty)`  (higher is better)

### Classification Effect (10 cases)

- best threshold: `3.0812`
- accuracy: `0.8000` (8/10)
- precision (good): `0.7143`
- recall (good): `1.0000`
- AUC: `0.8000`
- confusion: `TP=5, TN=3, FP=2, FN=0`
