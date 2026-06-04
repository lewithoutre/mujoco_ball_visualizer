#!/usr/bin/env python3
"""Live MuJoCo visualization for DVision ball and robot ROS topics."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "scene" / "ball_field.xml"
IDENTITY = np.eye(3).reshape(-1)
UNIT_SCALE = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}


@dataclass
class LiveState:
    raw_pos: Optional[np.ndarray] = None
    filtered_pos: Optional[np.ndarray] = None
    robot_pose: Optional[tuple[float, float, float]] = None
    raw_history: deque[np.ndarray] = field(default_factory=deque)
    filtered_history: deque[np.ndarray] = field(default_factory=deque)
    last_raw_stamp: float = 0.0
    last_filtered_stamp: float = 0.0
    last_vision_stamp: float = 0.0


def load_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        if exc.name == "mujoco":
            raise SystemExit("Python package 'mujoco' is not installed.") from exc
        raise
    return mujoco, mujoco.viewer


def load_ros():
    try:
        import rospy
        from dmsgs.msg import VisionInfo
        from geometry_msgs.msg import PointStamped
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ROS Python modules are not available. Source ROS and the robot workspace first:\n"
            "  source /opt/ros/noetic/setup.bash\n"
            "  source /home/lewithoutre/ZJU_dancer/robocup_ws/core/devel/setup.bash"
        ) from exc
    return rospy, VisionInfo, PointStamped


def valid_pos(pos: Optional[np.ndarray]) -> bool:
    return pos is not None and bool(np.isfinite(pos).all())


def point_msg_to_pos(msg, scale: float, ball_radius_m: float) -> np.ndarray:
    pos = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float) * scale
    if not math.isfinite(pos[2]) or pos[2] < ball_radius_m:
        pos[2] = ball_radius_m
    return pos


def vector_to_pos(vec, scale: float, ball_radius_m: float) -> np.ndarray:
    pos = np.array([vec.x, vec.y, vec.z], dtype=float) * scale
    if not math.isfinite(pos[2]) or pos[2] < ball_radius_m:
        pos[2] = ball_radius_m
    return pos


def normalize_yaw(value: float, yaw_unit: str) -> float:
    if yaw_unit == "deg":
        return math.radians(value)
    if yaw_unit == "rad":
        return value
    if abs(value) > 2.0 * math.pi + 0.25:
        return math.radians(value)
    return value


def set_freejoint_pose(mujoco, model, data, joint_name: str, pos: np.ndarray, yaw: float = 0.0) -> None:
    try:
        qpos = data.joint(joint_name).qpos
    except KeyError:
        return
    qpos[:3] = pos
    qpos[3:] = np.array([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)])
    mujoco.mj_forward(model, data)


def configure_model(model, ball_radius_m: float) -> None:
    model.geom("ball_geom").size[0] = ball_radius_m


def configure_camera(viewer) -> None:
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.2])
    viewer.cam.distance = 7.5
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -32


def add_line(mujoco, scn, start: np.ndarray, end: np.ndarray, width: float, rgba) -> bool:
    if scn.ngeom >= len(scn.geoms):
        return False
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        IDENTITY,
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    scn.ngeom += 1
    return True


def add_sphere(mujoco, scn, pos: np.ndarray, radius: float, rgba) -> bool:
    if scn.ngeom >= len(scn.geoms):
        return False
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos,
        IDENTITY,
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return True


def draw_history(mujoco, scn, history, width: float, rgba_start, rgba_end) -> bool:
    if len(history) < 2:
        return True
    count = len(history) - 1
    start = np.asarray(rgba_start, dtype=float)
    end = np.asarray(rgba_end, dtype=float)
    for i in range(count):
        alpha = i / max(count - 1, 1)
        rgba = (1.0 - alpha) * start + alpha * end
        if not add_line(mujoco, scn, history[i], history[i + 1], width, rgba):
            return False
    return True


def draw_live_scene(
    mujoco,
    scn,
    raw_pos: Optional[np.ndarray],
    filtered_pos: Optional[np.ndarray],
    raw_history,
    filtered_history,
    ball_radius_m: float,
) -> None:
    scn.ngeom = 0
    if not draw_history(
        mujoco,
        scn,
        raw_history,
        2.0,
        [1.0, 0.10, 0.08, 0.08],
        [1.0, 0.10, 0.08, 0.45],
    ):
        return
    if not draw_history(
        mujoco,
        scn,
        filtered_history,
        4.0,
        [0.0, 0.78, 1.0, 0.12],
        [0.0, 0.78, 1.0, 0.85],
    ):
        return
    if valid_pos(raw_pos):
        if not add_sphere(mujoco, scn, raw_pos, ball_radius_m * 0.65, [1.0, 0.05, 0.03, 0.62]):
            return
    if valid_pos(filtered_pos):
        add_sphere(mujoco, scn, filtered_pos, ball_radius_m * 0.25, [0.0, 0.95, 1.0, 0.9])


def append_history(history: deque[np.ndarray], pos: np.ndarray, maxlen: int) -> None:
    if not valid_pos(pos):
        return
    if history and float(np.linalg.norm(history[-1] - pos)) < 1e-5:
        return
    history.append(pos.copy())
    while len(history) > maxlen:
        history.popleft()


def topic_or_none(value: str) -> Optional[str]:
    value = value.strip()
    if not value or value.lower() in ("none", "off", "disable", "disabled"):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo MJCF model path.")
    parser.add_argument("--robot-id", type=int, default=1, help="Robot id used to build default topic names.")
    parser.add_argument("--vision-topic", default=None, help="VisionInfo topic. Default: /dvision_<id>/VisionInfo")
    parser.add_argument("--raw-topic", default=None, help="Raw ball PointStamped topic. Default: /dvision_<id>/ball_raw")
    parser.add_argument("--filtered-topic", default=None, help="Filtered ball PointStamped topic. Default: /dvision_<id>/ball_filtered")
    parser.add_argument("--unit", choices=sorted(UNIT_SCALE), default="cm", help="Incoming position unit.")
    parser.add_argument("--yaw-unit", choices=("auto", "deg", "rad"), default="auto", help="robot_pos.z unit.")
    parser.add_argument("--ball-radius-m", type=float, default=0.07, help="Visual ball radius in meters.")
    parser.add_argument("--trail-points", type=int, default=240, help="History length for each trajectory.")
    parser.add_argument("--stale-sec", type=float, default=1.0, help="Hide live debug points after this many seconds without updates.")
    parser.add_argument("--show-ui", action=argparse.BooleanOptionalAction, default=False, help="Show MuJoCo side panels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ball_radius_m <= 0.0:
        raise SystemExit("--ball-radius-m must be positive")
    if args.trail_points <= 0:
        raise SystemExit("--trail-points must be positive")

    topic_prefix = f"/dvision_{args.robot_id}"
    vision_topic = topic_or_none(args.vision_topic or f"{topic_prefix}/VisionInfo")
    raw_topic = topic_or_none(args.raw_topic or f"{topic_prefix}/ball_raw")
    filtered_topic = topic_or_none(args.filtered_topic or f"{topic_prefix}/ball_filtered")
    scale = UNIT_SCALE[args.unit]

    mujoco, viewer_module = load_mujoco()
    rospy, VisionInfo, PointStamped = load_ros()

    state = LiveState()
    lock = threading.Lock()

    def on_raw(msg) -> None:
        pos = point_msg_to_pos(msg, scale, args.ball_radius_m)
        with lock:
            state.raw_pos = pos
            state.last_raw_stamp = time.monotonic()
            append_history(state.raw_history, pos, args.trail_points)

    def on_filtered(msg) -> None:
        pos = point_msg_to_pos(msg, scale, args.ball_radius_m)
        with lock:
            state.filtered_pos = pos
            state.last_filtered_stamp = time.monotonic()
            append_history(state.filtered_history, pos, args.trail_points)

    def on_vision(msg) -> None:
        robot_x = msg.robot_pos.x * scale
        robot_y = msg.robot_pos.y * scale
        robot_yaw = normalize_yaw(msg.robot_pos.z, args.yaw_unit)
        now = time.monotonic()
        with lock:
            state.robot_pose = (robot_x, robot_y, robot_yaw)
            state.last_vision_stamp = now
            if filtered_topic is None and getattr(msg, "see_ball", False):
                pos = vector_to_pos(msg.ball_global, scale, args.ball_radius_m)
                state.filtered_pos = pos
                state.last_filtered_stamp = now
                append_history(state.filtered_history, pos, args.trail_points)

    rospy.init_node("mujoco_ball_live_viewer", anonymous=True, disable_signals=True)
    if vision_topic is not None:
        rospy.Subscriber(vision_topic, VisionInfo, on_vision, queue_size=1)
    if raw_topic is not None:
        rospy.Subscriber(raw_topic, PointStamped, on_raw, queue_size=1)
    if filtered_topic is not None:
        rospy.Subscriber(filtered_topic, PointStamped, on_filtered, queue_size=1)

    print("MuJoCo live viewer subscribing to:")
    print(f"  vision:   {vision_topic or 'disabled'}")
    print(f"  raw:      {raw_topic or 'disabled'}")
    print(f"  filtered: {filtered_topic or 'disabled'}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    configure_model(model, args.ball_radius_m)
    set_freejoint_pose(mujoco, model, data, "ball_freejoint", np.array([0.0, 0.0, args.ball_radius_m]))
    set_freejoint_pose(mujoco, model, data, "robot_freejoint", np.array([0.0, 0.0, 0.0]))

    with viewer_module.launch_passive(
        model,
        data,
        show_left_ui=args.show_ui,
        show_right_ui=args.show_ui,
    ) as viewer:
        with viewer.lock():
            configure_camera(viewer)
        viewer.sync()

        while viewer.is_running() and not rospy.is_shutdown():
            now = time.monotonic()
            with lock:
                raw_pos = state.raw_pos.copy() if valid_pos(state.raw_pos) else None
                filtered_pos = state.filtered_pos.copy() if valid_pos(state.filtered_pos) else None
                raw_history = list(state.raw_history)
                filtered_history = list(state.filtered_history)
                robot_pose = state.robot_pose
                raw_age = now - state.last_raw_stamp if state.last_raw_stamp else float("inf")
                filtered_age = now - state.last_filtered_stamp if state.last_filtered_stamp else float("inf")

            if raw_age > args.stale_sec:
                raw_pos = None
            if filtered_age > args.stale_sec:
                filtered_pos = None

            with viewer.lock():
                if valid_pos(filtered_pos):
                    set_freejoint_pose(mujoco, model, data, "ball_freejoint", filtered_pos)
                if robot_pose is not None:
                    x, y, yaw = robot_pose
                    set_freejoint_pose(mujoco, model, data, "robot_freejoint", np.array([x, y, 0.0]), yaw)
                draw_live_scene(
                    mujoco,
                    viewer.user_scn,
                    raw_pos,
                    filtered_pos,
                    raw_history,
                    filtered_history,
                    args.ball_radius_m,
                )

            viewer.sync()
            time.sleep(1.0 / 120.0)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
