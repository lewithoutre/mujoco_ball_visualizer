#!/usr/bin/env python3
"""Build a MuJoCo field scene that uses the full PiPlus MJCF robot."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD = ROOT / "scene" / "ball_field.xml"
DEFAULT_PIPLUS = ROOT.parent / "PiPlus" / "robot.xml"
DEFAULT_OUTPUT = ROOT / "scene" / "piplus_ball_field.xml"

SIMPLE_ROBOT_BODY = "robot"
VISUAL_GEOM_CLASS = "visual"
LEFT_FOOT_SITE = "lfoot-vismarker"
RIGHT_FOOT_SITE = "rfoot-vismarker"


def indent(element: ET.Element, level: int = 0) -> None:
    space = "\n" + level * "  "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = space + "  "

        for child in element:
            indent(child, level + 1)

        if not child.tail or not child.tail.strip():
            child.tail = space

    if level and (not element.tail or not element.tail.strip()):
        element.tail = space


def required(parent: ET.Element, name: str, source: Path) -> ET.Element:
    element = parent.find(name)

    if element is None:
        raise SystemExit(f"{source} does not contain <{name}>")

    return element


def remove_simple_robot(worldbody: ET.Element) -> None:
    for body in list(worldbody.findall("body")):
        if body.get("name") == SIMPLE_ROBOT_BODY:
            worldbody.remove(body)
            return

    raise SystemExit(f"Field scene does not contain body '{SIMPLE_ROBOT_BODY}'")


def parse_vec(value: str | None) -> list[float]:
    if value is None:
        return [0.0, 0.0, 0.0]

    parts = [float(part) for part in value.split()]

    if len(parts) != 3:
        raise SystemExit(f"Expected a 3D vector, got: {value}")

    return parts


def format_vec(values: list[float]) -> str:
    return " ".join(f"{value:.8g}" for value in values)


def find_site_pos(body: ET.Element, site_name: str, parent_pos: list[float] | None = None) -> list[float] | None:
    if parent_pos is None:
        parent_pos = [0.0, 0.0, 0.0]

    body_pos = parse_vec(body.get("pos"))
    current_pos = [parent_pos[i] + body_pos[i] for i in range(3)]

    for site in body.findall("site"):
        if site.get("name") == site_name:
            site_pos = parse_vec(site.get("pos"))
            return [current_pos[i] + site_pos[i] for i in range(3)]

    for child in body.findall("body"):
        found = find_site_pos(child, site_name, current_pos)

        if found is not None:
            return found

    return None


def foot_midpoint_xy(body: ET.Element) -> tuple[float, float]:
    left = find_site_pos(body, LEFT_FOOT_SITE)
    right = find_site_pos(body, RIGHT_FOOT_SITE)

    if left is None or right is None:
        raise SystemExit(
            f"PiPlus model must contain sites '{LEFT_FOOT_SITE}' and '{RIGHT_FOOT_SITE}'"
        )

    return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)


def remove_joints(body: ET.Element) -> None:
    for parent in body.iter():
        for child in list(parent):
            if child.tag in ("joint", "freejoint"):
                parent.remove(child)


def visual_defaults_only(defaults: ET.Element) -> ET.Element:
    defaults = copy.deepcopy(defaults)

    for child in list(defaults):
        if child.tag != "default":
            defaults.remove(child)

    return defaults


def wrap_static_robot(piplus_body: ET.Element) -> ET.Element:
    midpoint_x, midpoint_y = foot_midpoint_xy(piplus_body)
    torso_pos = parse_vec(piplus_body.get("pos"))

    # The live pose is the ground-frame x/y/yaw of the midpoint between both feet.
    piplus_body.set(
        "pos",
        format_vec([torso_pos[0] - midpoint_x, torso_pos[1] - midpoint_y, torso_pos[2]]),
    )

    remove_joints(piplus_body)

    robot_body = ET.Element("body", {"name": SIMPLE_ROBOT_BODY, "pos": "0 0 0"})
    ET.SubElement(robot_body, "freejoint", {"name": "robot_freejoint"})
    robot_body.append(piplus_body)

    return robot_body


def set_visual_geoms_non_colliding(body: ET.Element) -> None:
    for geom in body.iter("geom"):
        if geom.get("class") == VISUAL_GEOM_CLASS:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def build_scene(field_path: Path, piplus_path: Path, output_path: Path) -> None:
    field_tree = ET.parse(field_path)
    robot_tree = ET.parse(piplus_path)

    field_root = field_tree.getroot()
    robot_root = robot_tree.getroot()

    field_root.set("model", "dvision_piplus_ball_field")

    compiler = required(field_root, "compiler", field_path)
    compiler.set("meshdir", str(piplus_path.parent / "meshes"))

    field_asset = required(field_root, "asset", field_path)
    robot_asset = required(robot_root, "asset", piplus_path)

    for child in list(robot_asset):
        field_asset.append(copy.deepcopy(child))

    robot_default = required(robot_root, "default", piplus_path)
    field_root.insert(list(field_root).index(field_asset), visual_defaults_only(robot_default))

    field_worldbody = required(field_root, "worldbody", field_path)
    robot_worldbody = required(robot_root, "worldbody", piplus_path)
    piplus_body = robot_worldbody.find("body")

    if piplus_body is None:
        raise SystemExit(f"{piplus_path} does not contain a top-level robot body")

    piplus_body = copy.deepcopy(piplus_body)
    piplus_body = wrap_static_robot(piplus_body)
    set_visual_geoms_non_colliding(piplus_body)

    remove_simple_robot(field_worldbody)
    field_worldbody.append(piplus_body)

    indent(field_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_tree.write(output_path, encoding="utf-8", xml_declaration=False)
    output_path.write_text(output_path.read_text() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD, help="Base field MJCF path.")
    parser.add_argument("--piplus", type=Path, default=DEFAULT_PIPLUS, help="PiPlus robot MJCF path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated MJCF output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_scene(args.field, args.piplus, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
