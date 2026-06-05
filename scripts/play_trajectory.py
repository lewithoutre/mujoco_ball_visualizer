#!/usr/bin/env python3
"""Replay a ball trajectory CSV in a MuJoCo raw GLFW window."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import glfw
import numpy as np

from mujoco_glfw_controls import camera_controls_help, install_mouse_camera_controls


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "scene" / "piplus_ball_field.xml"
IDENTITY = np.eye(3).reshape(-1)

UNIT_SCALE = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}

AUTO_TIME_COLUMNS = ("time", "t", "stamp", "stamp_sec", "timestamp", "sec")


@dataclass(frozen=True)
class TrajectoryPoint:
    t: float
    pos: np.ndarray
    robot_pose: Optional[np.ndarray] = None


def load_mujoco_after_glfw_window():
    """
    Jetson workaround:
    create the GLFW window first, then import/use MuJoCo.

    Importing MuJoCo before creating the GLFW window may cause:
      GLX: No GLXFBConfigs returned
    """
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        if exc.name == "mujoco":
            raise SystemExit(
                "Python package 'mujoco' is not installed. Run:\n"
                "  python3 -m pip install mujoco"
            ) from exc
        raise

    return mujoco


def add_bool_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    """
    Python 3.8 compatible replacement for argparse.BooleanOptionalAction.

    Example:
      --loop
      --no-loop
    """
    if not name.startswith("--"):
        raise ValueError("Boolean argument name must start with '--'")

    dest = name[2:].replace("-", "_")
    no_name = "--no-" + name[2:]

    parser.add_argument(
        name,
        dest=dest,
        action="store_true",
        default=default,
        help=help_text,
    )
    parser.add_argument(
        no_name,
        dest=dest,
        action="store_false",
        help=f"Disable: {help_text}",
    )


def parse_column_list(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if len(columns) not in (2, 3):
        raise argparse.ArgumentTypeError("--columns must contain 2 or 3 comma-separated names")
    return columns


def parse_float(value: Optional[str]) -> float:
    if value is None:
        return math.nan

    value = value.strip()
    if not value:
        return math.nan

    try:
        return float(value)
    except ValueError:
        return math.nan


def resolve_column(fieldnames: Sequence[str], requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None

    stripped = {name.strip(): name for name in fieldnames}
    if requested in stripped:
        return stripped[requested]

    lowered = {name.strip().lower(): name for name in fieldnames}
    return lowered.get(requested.lower())


def resolve_auto_time_column(fieldnames: Sequence[str]) -> Optional[str]:
    for candidate in AUTO_TIME_COLUMNS:
        resolved = resolve_column(fieldnames, candidate)
        if resolved is not None:
            return resolved
    return None


def set_freejoint_pose(
    mujoco,
    model,
    data,
    joint_name: str,
    pos: np.ndarray,
    yaw: float = 0.0,
) -> None:
    try:
        qpos = data.joint(joint_name).qpos
    except KeyError:
        return

    qpos[:3] = pos
    qpos[3:] = np.array(
        [
            math.cos(yaw * 0.5),
            0.0,
            0.0,
            math.sin(yaw * 0.5),
        ],
        dtype=float,
    )

    mujoco.mj_forward(model, data)


def load_csv_trajectory(
    path: Path,
    columns: Sequence[str],
    robot_columns: Optional[Sequence[str]],
    time_column: Optional[str],
    fps: float,
    unit: str,
    ball_radius_m: float,
    clamp_ground: bool,
) -> list[TrajectoryPoint]:
    scale = UNIT_SCALE[unit]
    points: list[TrajectoryPoint] = []

    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        resolved_columns = [resolve_column(reader.fieldnames, col) for col in columns]
        missing = [columns[i] for i, resolved in enumerate(resolved_columns) if resolved is None]
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
            
        resolved_robot_columns = None
        if robot_columns is not None:
            resolved_robot_columns = [resolve_column(reader.fieldnames, col) for col in robot_columns]
            if any(c is None for c in resolved_robot_columns):
                resolved_robot_columns = None

        resolved_time = resolve_column(reader.fieldnames, time_column)
        if time_column and resolved_time is None:
            raise ValueError(f"Missing time column: {time_column}")

        if resolved_time is None:
            resolved_time = resolve_auto_time_column(reader.fieldnames)

        for index, row in enumerate(reader):
            values = [parse_float(row.get(col)) for col in resolved_columns if col is not None]
            if any(math.isnan(value) for value in values):
                continue
                
            robot_values = None
            if resolved_robot_columns is not None:
                parsed_robot = [parse_float(row.get(col)) for col in resolved_robot_columns if col is not None]
                if not any(math.isnan(v) for v in parsed_robot):
                    if len(parsed_robot) == 2:
                        parsed_robot.append(0.0) # default yaw
                    robot_values = np.array(parsed_robot, dtype=float)
                    robot_values[:2] *= scale

            if len(values) == 2:
                values.append(ball_radius_m / scale)

            pos = np.array(values, dtype=float) * scale

            if clamp_ground and pos[2] < ball_radius_m:
                pos[2] = ball_radius_m

            if resolved_time is not None:
                stamp = parse_float(row.get(resolved_time))
                if math.isnan(stamp):
                    continue
                t = stamp
            else:
                t = index / fps

            points.append(TrajectoryPoint(t=t, pos=pos, robot_pose=robot_values))

    return normalize_trajectory(points)


def normalize_trajectory(points: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
    if not points:
        return []

    points = sorted(points, key=lambda point: point.t)
    origin = points[0].t
    normalized: list[TrajectoryPoint] = []

    for point in points:
        shifted = point.t - origin
        if normalized and shifted <= normalized[-1].t:
            normalized[-1] = TrajectoryPoint(t=shifted, pos=point.pos, robot_pose=point.robot_pose)
        else:
            normalized.append(TrajectoryPoint(t=shifted, pos=point.pos, robot_pose=point.robot_pose))

    return normalized


def demo_trajectory(ball_radius_m: float) -> list[TrajectoryPoint]:
    dt = 1.0 / 60.0
    gravity = -9.81
    restitution = 0.52
    drag = 0.985

    pos = np.array([-3.6, -0.85, ball_radius_m], dtype=float)
    vel = np.array([1.60, 0.38, 3.60], dtype=float)

    points: list[TrajectoryPoint] = []

    for i in range(int(6.0 / dt) + 1):
        t = i * dt
        points.append(TrajectoryPoint(t=t, pos=pos.copy()))

        vel[2] += gravity * dt
        pos += vel * dt
        vel[:2] *= drag

        if pos[2] < ball_radius_m:
            pos[2] = ball_radius_m + (ball_radius_m - pos[2])
            vel[2] = -vel[2] * restitution
            vel[:2] *= 0.86

    return points


def sample_trajectory(
    points: Sequence[TrajectoryPoint],
    times: Sequence[float],
    t: float,
) -> tuple[np.ndarray, Optional[np.ndarray], int]:
    if t <= times[0]:
        return points[0].pos.copy(), (points[0].robot_pose.copy() if points[0].robot_pose is not None else None), 0

    if t >= times[-1]:
        return points[-1].pos.copy(), (points[-1].robot_pose.copy() if points[-1].robot_pose is not None else None), len(points) - 1

    right = bisect.bisect_right(times, t)
    left = right - 1

    t0 = times[left]
    t1 = times[right]

    alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    pos = (1.0 - alpha) * points[left].pos + alpha * points[right].pos
    
    robot_pose = None
    if points[left].robot_pose is not None and points[right].robot_pose is not None:
        robot_pose = (1.0 - alpha) * points[left].robot_pose + alpha * points[right].robot_pose

    return pos, robot_pose, left


def set_ball_pose(mujoco, model, data, pos: np.ndarray) -> None:
    qpos = data.joint("ball_freejoint").qpos
    qpos[:3] = pos
    qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)


def configure_model(model, ball_radius_m: float) -> None:
    model.geom("ball_geom").size[0] = ball_radius_m


def configure_camera(
    cam,
    points: Sequence[TrajectoryPoint],
    default_robot_pose: np.ndarray,
) -> None:
    positions = [point.pos for point in points]

    robot_positions = [
        np.array([point.robot_pose[0], point.robot_pose[1], 0.4], dtype=float)
        for point in points
        if point.robot_pose is not None
    ]

    if robot_positions:
        positions.extend(robot_positions)
    else:
        positions.append(np.array([default_robot_pose[0], default_robot_pose[1], 0.4], dtype=float))

    positions = np.array(positions)

    low = positions.min(axis=0)
    high = positions.max(axis=0)

    center = (low + high) * 0.5
    center[2] = max(center[2], 0.15)

    extent = max(float((high - low).max()), 1.0)

    cam.lookat[:] = center
    cam.distance = max(3.0, extent * 1.7)
    cam.azimuth = 140
    cam.elevation = -28


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

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        start,
        end,
    )

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


def draw_trail_scene(
    mujoco,
    scn,
    history: Sequence[np.ndarray],
    line_width: float,
    ghost_every: int,
) -> None:
    """
    Append trajectory trail geoms to the existing MuJoCo scene.

    Do not reset scn.ngeom here. mjv_updateScene has already filled
    the scene with the model geoms for this frame.
    """
    if len(history) < 2:
        return

    segment_count = len(history) - 1

    for i in range(segment_count):
        age = i / max(segment_count - 1, 1)

        rgba = [
            0.05 + 0.20 * age,
            0.75,
            1.00 - 0.25 * age,
            0.18 + 0.62 * age,
        ]

        if not add_line(mujoco, scn, history[i], history[i + 1], line_width, rgba):
            return

    if ghost_every <= 0:
        return

    for i in range(0, len(history), ghost_every):
        age = i / max(len(history) - 1, 1)
        rgba = [1.0, 0.92, 0.18, 0.18 + 0.35 * age]

        if not add_sphere(mujoco, scn, history[i], 0.025, rgba):
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("csv", nargs="?", type=Path, help="Trajectory CSV path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo MJCF model path.")

    parser.add_argument(
        "--columns",
        type=parse_column_list,
        default=parse_column_list("x,y,z"),
        help="Position columns, e.g. x,y,z or kf_x,kf_y,kf_z.",
    )

    parser.add_argument(
        "--robot-columns",
        type=parse_column_list,
        default=None,
        help="Robot position columns, e.g. robot_x,robot_y,robot_yaw. Must be 2 or 3 columns.",
    )

    parser.add_argument("--robot-x", type=float, default=-1.0, help="Default robot X if not in CSV.")
    parser.add_argument("--robot-y", type=float, default=0.0, help="Default robot Y if not in CSV.")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Default robot yaw (radians) if not in CSV.")

    parser.add_argument("--time-column", default=None, help="Optional timestamp column.")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS when no time column exists.")
    parser.add_argument("--unit", choices=sorted(UNIT_SCALE), default="cm", help="CSV position unit.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier.")
    parser.add_argument("--ball-radius-m", type=float, default=0.07, help="Visual ball radius in meters.")
    parser.add_argument("--trail-points", type=int, default=240, help="Number of historical samples to draw.")
    parser.add_argument("--line-width", type=float, default=4.0, help="Trail line width in pixels.")

    parser.add_argument(
        "--ghost-every",
        type=int,
        default=20,
        help="Draw a ghost point every N trail points; 0 disables it.",
    )

    parser.add_argument("--width", type=int, default=1000, help="Window width.")
    parser.add_argument("--height", type=int, default=700, help="Window height.")

    add_bool_argument(parser, "--loop", default=True, help_text="Loop playback.")
    add_bool_argument(
        parser,
        "--show-ui",
        default=False,
        help_text="Show MuJoCo side panels. Ignored in raw GLFW mode.",
    )
    add_bool_argument(
        parser,
        "--clamp-ground",
        default=True,
        help_text="Clamp z below the ball radius to the ground.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Validate trajectory loading without opening MuJoCo.")

    return parser.parse_args()


def key_callback(window, key, scancode, action, mods) -> None:
    del scancode, mods

    if action != glfw.PRESS:
        return

    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)


def create_glfw_window(width: int, height: int):
    if not glfw.init():
        raise SystemExit("ERROR: glfw.init() failed")

    window = glfw.create_window(width, height, "MuJoCo trajectory player", None, None)

    if not window:
        glfw.terminate()
        raise SystemExit(
            "ERROR: glfw.create_window() failed\n"
            "Try running:\n"
            "  export DISPLAY=:0\n"
            "  unset MUJOCO_GL\n"
            "  unset PYOPENGL_PLATFORM\n"
            "  unset LIBGL_ALWAYS_INDIRECT\n"
        )

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, key_callback)

    return window


def main() -> int:
    args = parse_args()

    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive")

    if args.speed <= 0.0:
        raise SystemExit("--speed must be positive")

    if args.ball_radius_m <= 0.0:
        raise SystemExit("--ball-radius-m must be positive")

    if args.csv is None:
        points = demo_trajectory(args.ball_radius_m)
        source = "built-in demo"
    else:
        points = load_csv_trajectory(
            args.csv,
            args.columns,
            args.robot_columns,
            args.time_column,
            args.fps,
            args.unit,
            args.ball_radius_m,
            args.clamp_ground,
        )
        source = str(args.csv)

    if not points:
        raise SystemExit("No valid trajectory points were loaded.")

    times = [point.t for point in points]
    duration = max(times[-1], 1e-9)

    unit_label = args.unit if args.csv is not None else "m"

    summary = (
        f"Loaded {len(points)} points from {source}; "
        f"duration={times[-1]:.3f}s, unit={unit_label}."
    )

    if args.dry_run:
        print(summary)
        return 0

    if args.show_ui:
        print("Note: --show-ui is ignored in raw GLFW mode.")

    print(summary)
    print("Creating GLFW window before importing MuJoCo...")

    window = create_glfw_window(args.width, args.height)

    mujoco = load_mujoco_after_glfw_window()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    configure_model(model, args.ball_radius_m)
    set_ball_pose(mujoco, model, data, points[0].pos)

    robot_pose = points[0].robot_pose
    if robot_pose is None:
        set_freejoint_pose(
            mujoco,
            model,
            data,
            "robot_freejoint",
            np.array([args.robot_x, args.robot_y, 0.0], dtype=float),
            args.robot_yaw,
        )
    else:
        rx, ry = robot_pose[0], robot_pose[1]
        ryaw = robot_pose[2] if len(robot_pose) > 2 else 0.0
        set_freejoint_pose(
            mujoco,
            model,
            data,
            "robot_freejoint",
            np.array([rx, ry, 0.0], dtype=float),
            ryaw,
        )

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    default_robot_pose = np.array([args.robot_x, args.robot_y, args.robot_yaw], dtype=float)
    configure_camera(cam, points, default_robot_pose)
    mouse_controller = install_mouse_camera_controls(window, mujoco, model, scene, cam)

    start_wall = time.monotonic()
    last_print = 0.0

    print("Running. Close the window or press ESC to stop.")
    print(camera_controls_help())

    try:
        while not glfw.window_should_close(window):
            elapsed = (time.monotonic() - start_wall) * args.speed

            if args.loop:
                replay_t = elapsed % duration
            else:
                replay_t = min(elapsed, duration)

            pos, current_robot_pose, index = sample_trajectory(points, times, replay_t)

            start_index = max(0, index - max(args.trail_points, 0))
            history = [point.pos for point in points[start_index : index + 1]]
            history.append(pos)

            set_ball_pose(mujoco, model, data, pos)

            if current_robot_pose is not None:
                rx, ry = current_robot_pose[0], current_robot_pose[1]
                ryaw = current_robot_pose[2] if len(current_robot_pose) > 2 else 0.0
                set_freejoint_pose(
                    mujoco,
                    model,
                    data,
                    "robot_freejoint",
                    np.array([rx, ry, 0.0], dtype=float),
                    ryaw,
                )

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            draw_trail_scene(
                mujoco,
                scene,
                history,
                args.line_width,
                args.ghost_every,
            )

            mujoco.mjr_render(viewport, scene, context)

            glfw.swap_buffers(window)
            glfw.poll_events()

            now = time.monotonic()
            if now - last_print > 1.0:
                print(
                    f"\rt={replay_t:6.3f}s  "
                    f"pos=({pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}) m",
                    end="",
                    flush=True,
                )
                last_print = now

            time.sleep(1.0 / 120.0)

    finally:
        print()
        glfw.terminate()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        try:
            glfw.terminate()
        except Exception:
            pass
        sys.exit(130)
