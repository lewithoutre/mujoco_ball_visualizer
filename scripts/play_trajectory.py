#!/usr/bin/env python3
"""Replay a ball trajectory CSV in the MuJoCo viewer."""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "scene" / "ball_field.xml"
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


def load_mujoco():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        if exc.name == "mujoco":
            raise SystemExit(
                "Python package 'mujoco' is not installed. Run:\n"
                "  cd mujoco_ball_visualizer\n"
                "  python3 -m venv .venv\n"
                "  source .venv/bin/activate\n"
                "  pip install -r requirements.txt"
            ) from exc
        raise
    return mujoco, mujoco.viewer


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


def load_csv_trajectory(
    path: Path,
    columns: Sequence[str],
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

        resolved_time = resolve_column(reader.fieldnames, time_column)
        if time_column and resolved_time is None:
            raise ValueError(f"Missing time column: {time_column}")
        if resolved_time is None:
            resolved_time = resolve_auto_time_column(reader.fieldnames)

        for index, row in enumerate(reader):
            values = [parse_float(row.get(col)) for col in resolved_columns if col is not None]
            if any(math.isnan(value) for value in values):
                continue

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
            points.append(TrajectoryPoint(t=t, pos=pos))

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
            normalized[-1] = TrajectoryPoint(t=shifted, pos=point.pos)
        else:
            normalized.append(TrajectoryPoint(t=shifted, pos=point.pos))
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
) -> tuple[np.ndarray, int]:
    if t <= times[0]:
        return points[0].pos.copy(), 0
    if t >= times[-1]:
        return points[-1].pos.copy(), len(points) - 1

    right = bisect.bisect_right(times, t)
    left = right - 1
    t0 = times[left]
    t1 = times[right]
    alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    pos = (1.0 - alpha) * points[left].pos + alpha * points[right].pos
    return pos, left


def set_ball_pose(mujoco, model, data, pos: np.ndarray) -> None:
    qpos = data.joint("ball_freejoint").qpos
    qpos[:3] = pos
    qpos[3:] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)


def configure_model(model, ball_radius_m: float) -> None:
    model.geom("ball_geom").size[0] = ball_radius_m


def configure_camera(viewer, points: Sequence[TrajectoryPoint]) -> None:
    positions = np.array([point.pos for point in points])
    low = positions.min(axis=0)
    high = positions.max(axis=0)
    center = (low + high) * 0.5
    center[2] = max(center[2], 0.15)
    extent = max(float((high - low).max()), 1.0)

    viewer.cam.lookat[:] = center
    viewer.cam.distance = max(3.0, extent * 1.7)
    viewer.cam.azimuth = 140
    viewer.cam.elevation = -28


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


def draw_user_scene(
    mujoco,
    scn,
    history: Sequence[np.ndarray],
    line_width: float,
    ghost_every: int,
) -> None:
    scn.ngeom = 0
    if len(history) < 2:
        return

    segment_count = len(history) - 1
    for i in range(segment_count):
        age = i / max(segment_count - 1, 1)
        rgba = [0.05 + 0.20 * age, 0.75, 1.00 - 0.25 * age, 0.18 + 0.62 * age]
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
    parser.add_argument("--time-column", default=None, help="Optional timestamp column.")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS when no time column exists.")
    parser.add_argument("--unit", choices=sorted(UNIT_SCALE), default="cm", help="CSV position unit.")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier.")
    parser.add_argument("--ball-radius-m", type=float, default=0.07, help="Visual ball radius in meters.")
    parser.add_argument("--trail-points", type=int, default=240, help="Number of historical samples to draw.")
    parser.add_argument("--line-width", type=float, default=4.0, help="Trail line width in pixels.")
    parser.add_argument("--ghost-every", type=int, default=20, help="Draw a ghost point every N trail points; 0 disables it.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Loop playback.")
    parser.add_argument("--show-ui", action=argparse.BooleanOptionalAction, default=False, help="Show MuJoCo side panels.")
    parser.add_argument("--dry-run", action="store_true", help="Validate trajectory loading without opening MuJoCo.")
    parser.add_argument(
        "--clamp-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp z below the ball radius to the ground.",
    )
    return parser.parse_args()


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

    mujoco, viewer_module = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    configure_model(model, args.ball_radius_m)

    print(
        f"{summary} Close the MuJoCo window to stop."
    )

    with viewer_module.launch_passive(
        model,
        data,
        show_left_ui=args.show_ui,
        show_right_ui=args.show_ui,
    ) as viewer:
        with viewer.lock():
            configure_camera(viewer, points)
            set_ball_pose(mujoco, model, data, points[0].pos)
        viewer.sync()

        start_wall = time.monotonic()
        last_print = 0.0
        while viewer.is_running():
            elapsed = (time.monotonic() - start_wall) * args.speed
            if args.loop:
                replay_t = elapsed % duration
            else:
                replay_t = min(elapsed, duration)

            pos, index = sample_trajectory(points, times, replay_t)
            start_index = max(0, index - max(args.trail_points, 0))
            history = [point.pos for point in points[start_index : index + 1]]
            history.append(pos)

            with viewer.lock():
                set_ball_pose(mujoco, model, data, pos)
                draw_user_scene(
                    mujoco,
                    viewer.user_scn,
                    history,
                    args.line_width,
                    args.ghost_every,
                )

            viewer.sync()

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

    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
