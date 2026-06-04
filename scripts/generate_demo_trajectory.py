#!/usr/bin/env python3
"""Generate a demo ball trajectory CSV in DVision-style centimeter units."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "examples" / "demo_trajectory.csv"


def simulate(duration: float, fps: float):
    dt = 1.0 / fps
    ball_radius = 7.0
    gravity = -981.0
    restitution = 0.52
    drag = 0.985

    pos = [-360.0, -85.0, ball_radius]
    vel = [160.0, 38.0, 360.0]
    t = 0.0

    while t <= duration + 1e-9:
        yield t, pos[0], pos[1], pos[2]

        vel[2] += gravity * dt
        pos[0] += vel[0] * dt
        pos[1] += vel[1] * dt
        pos[2] += vel[2] * dt
        vel[0] *= drag
        vel[1] *= drag

        if pos[2] < ball_radius:
            pos[2] = ball_radius + (ball_radius - pos[2])
            vel[2] = -vel[2] * restitution
            vel[0] *= 0.86
            vel[1] *= 0.86

        t += dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output path.",
    )
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in seconds.")
    parser.add_argument("--fps", type=float, default=60.0, help="Samples per second.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["time", "x", "y", "z"])
        for t, x, y, z in simulate(args.duration, args.fps):
            writer.writerow([f"{t:.4f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
