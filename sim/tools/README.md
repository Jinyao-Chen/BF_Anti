# sim/tools/ —— 一次性探针与标定工具

> 这里的脚本是**开发期一次性工具**：用来回答某个具体问题（符号、增益、极性、时延），
> 结论会被写进 `docs/` 或 `sim/results/` 后就不再维护。**不保证风格统一，也不保证还能跑。**

| 子目录 | 内容 |
|---|---|
| `legacy_xy_probes/` | **历史归档**：XY 速度外环阶段（2026-09）的 9 个探针，仅作追溯参考 |

## 约定

1. 新探针放本目录，文件名 `<主题>_probe.py`，首行 docstring 写**要回答的问题**与用法；
2. 探针依赖 `~/betaflight/src/platform/SIMULATOR/gazebo/scripts` 的 Gazebo 桥接库；
3. 结论不要只留在脚本里 —— 抄进 [../../sim/results/](../../sim/results/) 或 [../../docs/](../../docs/) 才算完成。

日常入口（不是探针）在上一层：[../README.md](../README.md)。
