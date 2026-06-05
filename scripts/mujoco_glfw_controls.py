"""Mouse camera controls for raw GLFW MuJoCo windows."""

from __future__ import annotations

import glfw


class MouseCameraController:
    def __init__(self, mujoco, model, scene, cam):
        self.mujoco = mujoco
        self.model = model
        self.scene = scene
        self.cam = cam
        self.last_x = None
        self.last_y = None

    def install(self, window) -> None:
        glfw.set_mouse_button_callback(window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(window, self.on_cursor_pos)
        glfw.set_scroll_callback(window, self.on_scroll)
        glfw.set_key_callback(window, self.on_key)

    def on_mouse_button(self, window, button: int, action: int, mods: int) -> None:
        del button, mods

        if action == glfw.PRESS:
            self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def on_cursor_pos(self, window, xpos: float, ypos: float) -> None:
        if self.last_x is None or self.last_y is None:
            self.last_x = xpos
            self.last_y = ypos
            return

        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if not (left or right or middle):
            return

        _, height = glfw.get_window_size(window)
        if height <= 0:
            return

        rel_x = dx / height
        rel_y = dy / height

        if shift and left:
            self.move(self.mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, rel_y)
        elif middle or (left and right):
            self.move(self.mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, rel_y)
        elif right:
            self.move(self.mujoco.mjtMouse.mjMOUSE_MOVE_H, rel_x, 0.0)
            self.move(self.mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.0, rel_y)
        else:
            self.move(self.mujoco.mjtMouse.mjMOUSE_ROTATE_H, rel_x, 0.0)
            self.move(self.mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0.0, rel_y)

    def on_scroll(self, window, xoffset: float, yoffset: float) -> None:
        del window, xoffset
        self.move(self.mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset)

    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        del scancode, mods

        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.move(self.mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.move(self.mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.05)

    def move(self, action, rel_x: float, rel_y: float) -> None:
        self.mujoco.mjv_moveCamera(
            self.model,
            action,
            rel_x,
            rel_y,
            self.scene,
            self.cam,
        )


def install_mouse_camera_controls(window, mujoco, model, scene, cam) -> MouseCameraController:
    controller = MouseCameraController(mujoco, model, scene, cam)
    controller.install(window)
    return controller


def camera_controls_help() -> str:
    return (
        "Mouse controls: left-drag rotate, right-drag pan, wheel zoom, "
        "Shift+left-drag zoom, +/- zoom."
    )
