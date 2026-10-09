# tests/ —— 单元测试（153 项）

```bash
python3 -m pytest -q                                          # 全量
python3 -m pytest -q tests/test_p_visual_guidance.py           # 只跑控制律
```

约定：`testpaths = tests`（见 [../pytest.ini](../pytest.ini)）；测试以**裸名**导入仓库根模块
（`crsf` / `msp` / `rc_mux` / `tracking_controller`）与**包名**导入 `guidance.*` / `perception.*` / `modeling.*`。

| 测试文件 | 覆盖 |
|---|---|
| `test_p_visual_guidance.py` | 当前主线控制律：P + 比例导引项、软饱和、门控、垂向串级 P，以及"禁止 I/D 与跨帧状态"的属性审计 |
| `test_p_strike_kinematic.py` | 制导-控制运动学包线（含"目标高于本机 25 m"等阻塞工况） |
| `test_bbox_tracker.py` | 仿真检测框跟踪（确认帧、外推、门控、fresh 语义） |
| `test_camera_geometry.py` / `test_target_estimator.py` | 相机几何与目标状态估计 |
| `test_rc_mux.py` / `test_crsf.py` / `test_msp_flags.py` | RC 融合、CRSF 解码、MSP 状态位 |
| `test_tracking_controller.py` | 机载 PBVS 伺服 / 状态机 / PID |
| `test_track_cmd_adapter.py` | 制导输出 -> `rc_mux.TrackCmd` 适配 |
| `test_theory_model.py` / `test_coupled_altitude.py` / `test_altitude_diagnostics.py` | 理论模型、高度环、掉高诊断 |
| `test_hover_throttle.py` / `test_lift_loss.py` / `test_firmware_params.py` | 悬停油门、掉升力辨识、固件参数一致性 |
| `test_rotor_force_bridge.py` / `test_rotor_force_calibration.py` | 旋翼力模型桥接与标定 |
| `test_reference_trajectory.py` | 闭环参考轨迹插值（相位 / 首帧问题） |
| `test_intercept_guidance.py` / `test_intercept_bridge.py` | 历史阶段的拦截制导与其桥接 |
| `test_visual_strike_guidance.py` / `test_visual_strike_kinematic.py` / `test_visual_strike_rc_mapping.py` | P 律之前的纯视觉打击链路（历史，仍在回归里） |

静态检查（提交前必过，当前零告警）：

```bash
python3 -m flake8 --select=E9,F63,F7,F82,F401,F841,E722,W605 .
```
