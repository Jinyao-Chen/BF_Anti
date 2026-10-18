# perception/ —— 相机几何与目标状态估计

纯计算层，被仿真链路（`sim/`）与 Pi5 机载链路（`onboard/`）共用。
旧 X5 的 `detect.py` 已归档到 `legacy/rdk_x5_pico/`。

| 文件 | 职责 |
|---|---|
| `camera_geometry.py` | 针孔成像模型：像素 <-> 视线角、单目测距 `r = f_x * W / s`、相机上仰角与视线仰角换算 |
| `target_estimator.py` | 由检测框 + 本机位姿估计目标 3D 状态（含目标高度 `z_t`） |
| `target_kf.py` | 3D 卡尔曼滤波（平滑框抖动 / 短时遮挡） |

## 参数单位（踩过坑）

- `f_x` / `f_y` / 主点 / 框尺寸 `s` 全部是**像素**；
- 相机上仰角 `theta_cam`（命令行 `--tilt`）是**度**；
- 单目测距的 `W` 是**探测器标定出的等效目标宽度**，不是物理跨距：换检测模型必须重标（同一架靶机，`.pt` 用 2.00、ONNX 用 2.38），标定方法见 `sim/eval_detector_frames.py`。

测试：`tests/test_camera_geometry.py`、`tests/test_target_estimator.py`。
