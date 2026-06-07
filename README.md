# MuJoCo Ball Trajectory Visualizer

这个目录是一个独立的球轨迹可视化小工具，不依赖当前 ROS/CMake 构建。它用 MuJoCo 打开一个足球场景，然后按 CSV 中的轨迹驱动球的位置，适合拿来观察原始测量值、Kalman 滤波结果或预测轨迹。

默认场景会从同级目录 `../PiPlus/robot.xml` 合成完整 PiPlus 机器人模型，不再用简单几何体代替机器人。
合成后的机器人是固定姿态，只保留一个 `robot_freejoint` 用于对齐全局 `x/y/yaw`；这个 freejoint 的原点定义为两脚中点在地面上的投影。

## 安装

```bash
cd mujoco_ball_visualizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前项目坐标大多是厘米，播放器默认 `--unit cm`，会自动换算成 MuJoCo 的米制坐标。

如果 `../PiPlus/robot.xml` 或 STL mesh 更新了，重新生成 MuJoCo 场景：

```bash
python3 scripts/build_piplus_scene.py
```

## 快速跑通

不传 CSV 时会播放一个内置示例轨迹：

```bash
python3 scripts/play_trajectory.py
```

窗口支持鼠标视角控制：左键/中键拖动旋转视角，右键拖动平移，滚轮缩放，`Shift` + 左键上下拖动缩放，`1` 俯视，`2` 平视，`3` 斜俯视，`R` 重置视角，`+/-` 缩放，`ESC` 退出。

也可以先生成一份示例 CSV，再按文件回放：

```bash
python3 scripts/generate_demo_trajectory.py
python3 scripts/play_trajectory.py examples/demo_trajectory.csv --columns x,y,z --time-column time --unit cm
```

## 实时 ROS 可视化

`dvision` 现在会额外发布两个球可视化 topic，并从 `VisionInfo` 读取机器人位姿：

- `/dvision_<robot_id>/VisionInfo`: 机器人全局位姿和常规视觉信息，依赖 `dmsgs`。
- `/dvision_<robot_id>/ball_raw`: 视觉原始测量球，全局场地坐标，单位 cm。
- `/dvision_<robot_id>/ball_filtered`: Kalman 滤波球，全局场地坐标，单位 cm。

启动 MuJoCo 实时窗口：

```bash
python3 scripts/live_from_ros.py --robot-id 1
```

实时窗口中，原始观测会显示为红色散点，Kalman 滤波结果会显示为蓝色散点。终端默认会输出相邻两个滤波点之间的时间间隔；如果输出太频繁，可以加 `--no-print-filtered-step` 关闭。
按 `C` 可以一键清空当前窗口里的红色/蓝色球轨迹，方便同一个终端和窗口里多次调试；按 `ESC` 退出。

## CSV 格式

推荐格式：

```csv
time,x,y,z
0.000,-350.0,-80.0,7.0
0.033,-345.2,-78.4,8.3
```

如果你目前只有 2D Kalman 输出，也可以只给 `x,y`，播放器会把 `z` 固定在球半径高度：

```bash
python3 scripts/play_trajectory.py /path/to/kalman.csv --columns x,y --fps 30 --unit cm
```

如果 CSV 中同时保存了原始值和滤波值，比如：

```csv
time,raw_x,raw_y,raw_z,kf_x,kf_y,kf_z
```

回放滤波轨迹：

```bash
python3 scripts/play_trajectory.py /path/to/ball.csv --columns kf_x,kf_y,kf_z --time-column time --unit cm
```

回放原始测量轨迹：

```bash
python3 scripts/play_trajectory.py /path/to/ball.csv --columns raw_x,raw_y,raw_z --time-column time --unit cm
```

## 常用参数

- `--columns x,y,z`: 位置列名，支持 2 列或 3 列。
- `--time-column time`: 时间列名。不提供时会自动找 `time/t/stamp/timestamp`；仍找不到则按 `--fps` 生成时间。
- `--unit cm`: 输入坐标单位，支持 `cm`、`m`、`mm`。
- `--speed 0.5`: 半速回放。
- `--no-loop`: 播放完停在最后一帧。
- `--trail-points 240`: 轨迹尾迹点数。
- `--ghost-every 20`: 每隔多少个历史点画一个半透明轨迹点，设为 `0` 关闭。
- `--dry-run`: 只检查 CSV 是否能解析，不打开 MuJoCo 窗口。

MuJoCo 官方 Python viewer 支持 `launch_passive`，所以脚本能一边更新数据一边保持交互视角。macOS 上如果使用 passive viewer，需要用 `mjpython` 运行脚本；Linux/Ubuntu 一般直接用 `python3` 即可。
