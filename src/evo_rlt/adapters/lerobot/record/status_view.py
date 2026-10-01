# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ported from the LeRobot 0.6 working copy (lerobot.utils.status_view); LeRobot 0.5.1 lacks it.
"""Live operator window for interactive recording.

An OpenCV window that tiles the robot's camera feeds side by side with a panel showing
task / subtask, recording state, error flag, cadence and dataset counters.  It answers the
question the operator actually has mid-episode — "is it recording, which subtask am I on,
how many frames do I have?" — which is why it is a separate switch from ``--display_data``
rather than another visualization backend: ``log_visualization_data`` carries only an
observation and an action, none of the session state drawn here.

Threading model
---------------
OpenCV's HighGUI only reliably maps and repaints a window from the **main thread**, so
this class splits the two halves:

* :meth:`start` (main thread) imports cv2, checks for a display, and creates the window.
* :meth:`update` may be called from any thread and only stashes the latest
  ``(images, status)`` under a lock — it never touches cv2.
* :meth:`render_once` must be called from the **main thread**; it draws the stashed data
  and pumps the GUI event loop.  It self-throttles to ``render_fps``, so a 30 Hz control
  loop calling it every tick still only repaints ~15 times a second.

Optional-dependency note: ``opencv-python-headless`` is a *core* dependency of this
package, so ``import cv2`` always succeeds and the usual ``_foo_available`` guard would
tell us nothing.  What a headless wheel lacks is the GUI backend, which can only be probed
at runtime — hence the lazy import plus a try/except around ``namedWindow``.  If anything
is missing the view degrades to a no-op (logged once) and recording continues.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["StatusView"]

# waitKey codes that have a canonical key name rather than a printable character, so a key
# typed into the window reaches the same state machine the keyboard listener drives.
_WAITKEY_NAMES = {27: "esc", 32: "space", 13: "enter", 9: "tab", 8: "backspace"}


def _is_image(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None:
        return False
    return len(shape) == 2 or (len(shape) == 3 and shape[2] in (1, 3, 4))


def shadowed_x11_libs(cv2: Any) -> list[str]:
    """xcb libraries opencv-python bundles but that were loaded from another wheel first.

    opencv-python and PyAV both ship ``libxcb-shm``/``libxcb-shape``/``libxcb-xfixes`` under
    identical hash-suffixed sonames, and the dynamic loader keeps whichever copy arrives
    first. PyAV's copies are linked against PyAV's private ``libxcb``, so when ``av`` is
    imported before ``cv2`` the Qt backend drives one X connection through two libxcb
    instances and blocks forever inside ``cv2.namedWindow``. Callers that want the window
    must therefore ``import cv2`` before anything imports ``av`` (``lerobot.datasets``).

    Linux-only; returns an empty list when nothing is shadowed or it cannot be determined.
    """
    site_packages = Path(cv2.__file__).resolve().parent.parent
    try:
        bundled = {
            lib.name: libs_dir
            for libs_dir in site_packages.glob("opencv*.libs")
            for lib in libs_dir.iterdir()
            if lib.name.startswith("libxcb")
        }
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return []
    shadowed = set()
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        mapped = Path(fields[5])
        libs_dir = bundled.get(mapped.name)
        if libs_dir is not None and mapped.parent != libs_dir:
            shadowed.add(str(mapped))
    return sorted(shadowed)


class StatusView:
    """OpenCV window showing camera feeds plus a recording-status overlay.

    Args:
        camera_names: Restricts and orders the image keys shown. ``None`` auto-detects
            every image-shaped value in the dict passed to :meth:`update`.
        bgr: ``True`` when incoming frames are RGB and need converting for cv2. Default
            ``True``, because LeRobot cameras emit ``ColorMode.RGB``.
        max_width: The tiled camera strip is downscaled to at most this width.
        window_name: Window title.
        render_fps: Upper bound on repaints per second; extra :meth:`render_once` calls
            return immediately.
        panel_height: Height in pixels of the status panel drawn under the cameras.
    """

    def __init__(
        self,
        camera_names: Sequence[str] | None = None,
        *,
        bgr: bool = True,
        max_width: int = 1280,
        window_name: str = "lerobot-record",
        render_fps: float = 15.0,
        panel_height: int = 300,
    ) -> None:
        self.camera_names = list(camera_names) if camera_names else None
        self.bgr = bgr
        self.max_width = max_width
        self.window_name = window_name
        self.frame_interval = 1.0 / max(1e-3, render_fps)
        self.panel_height = panel_height

        self._lock = threading.Lock()
        self._latest_images: dict[str, Any] | None = None
        self._latest_status: dict[str, Any] | None = None
        self._enabled = False
        self._window_created = False
        self._render_errors = 0
        self._last_render = 0.0
        self._cv2: Any = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> StatusView:
        """Set up cv2 and the window. MUST be called from the main thread.

        Always returns ``self``; on any failure the view is simply left disabled, so the
        caller never has to branch on whether a display was available.
        """
        from evo_rlt.adapters.lerobot.record.key_input import is_headless

        try:
            import cv2  # noqa: PLC0415
        except Exception as exc:
            logger.warning("Status view disabled: OpenCV unavailable (%s).", exc)
            return self

        if is_headless():
            logger.warning(
                "Status view disabled: no DISPLAY/WAYLAND_DISPLAY. Use a graphical session "
                "(or `ssh -X`) to get the operator window."
            )
            return self

        shadowed = shadowed_x11_libs(cv2)
        if shadowed:
            # Creating the window now would hang the control loop, not just the view.
            logger.warning(
                "Status view disabled: OpenCV's bundled X11 libraries were already loaded by another "
                "wheel (%s), which deadlocks cv2.namedWindow. Import cv2 before anything imports "
                "PyAV (`av`, pulled in by lerobot.datasets).",
                ", ".join(shadowed),
            )
            return self

        self._cv2 = cv2
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)  # pump the event loop so the window actually maps
        except Exception as exc:
            # Both wheels ship the same `cv2/` package, so they cannot sit side by side: the
            # headless one has to be swapped out for the GUI build of the same version.
            logger.warning(
                "Status view disabled: cannot create window '%s' (%s). This OpenCV build has no GUI "
                "backend (opencv-python-headless). Swap it for the GUI build: "
                "`pip uninstall -y opencv-python-headless && pip install \"opencv-python==%s.*\"`.",
                self.window_name,
                exc,
                cv2.__version__.split("-")[0],
            )
            self._cv2 = None
            return self

        self._window_created = True
        self._enabled = True
        logger.info("Status view started: window='%s' (rendered from the main thread)", self.window_name)
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, images: dict[str, Any] | None, status: dict[str, Any] | None) -> None:
        """Stash the latest images and status. Non-blocking; safe from any thread."""
        if not self._enabled:
            return
        with self._lock:
            self._latest_images = images
            self._latest_status = status

    def render_once(self, on_key: Callable[[str], None] | None = None) -> bool:
        """Draw the stashed frame and pump the GUI event loop. Main thread only.

        Self-throttling: returns ``True`` immediately when the previous repaint is more
        recent than ``1 / render_fps``, so this is cheap to call every control tick.

        Args:
            on_key: When given, a key typed into the window is forwarded as a canonical key
                name (the same vocabulary :mod:`lerobot.utils.keyboard_input` emits) and
                this always returns ``True``. When ``None``, ``q``/Esc instead returns
                ``False`` to tell the caller to shut down.

        Returns:
            ``False`` only when the operator asked to quit via the window and ``on_key``
            was not supplied; ``True`` otherwise, including when the view is disabled.
        """
        if not self._enabled:
            return True
        now = time.perf_counter()
        if now - self._last_render < self.frame_interval:
            return True
        self._last_render = now

        cv2 = self._cv2
        with self._lock:
            images = self._latest_images
            status = dict(self._latest_status) if self._latest_status else {}
        try:
            canvas = self._render(images, status)
            if canvas is not None:
                cv2.imshow(self.window_name, canvas)
            self._render_errors = 0
        except Exception as exc:
            self._render_errors += 1
            if self._render_errors <= 3:
                logger.warning("Status view render error: %s", exc)
        try:
            key = cv2.waitKey(1) & 0xFF
        except Exception:
            key = 0xFF

        if key == 0xFF:  # no key pressed
            return True
        if on_key is not None:
            name = _WAITKEY_NAMES.get(key)
            if name is None and 32 < key < 127:
                name = chr(key).lower()
            if name is not None:
                on_key(name)
            return True
        return key not in (ord("q"), 27)

    def stop(self) -> None:
        """Destroy the window. Idempotent; safe to call on a disabled view."""
        if not self._enabled and not self._window_created:
            return
        self._enabled = False
        try:
            if self._cv2 is not None:
                self._cv2.destroyWindow(self.window_name)
                self._cv2.waitKey(1)
        except Exception:
            logger.debug("Could not destroy status view window", exc_info=True)
        self._window_created = False

    # ------------------------------------------------------------------ drawing

    def _select_images(self, images: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
        if self.camera_names:
            return [
                (name, np.asarray(images[name]))
                for name in self.camera_names
                if name in images and _is_image(images[name])
            ]
        return [(key, np.asarray(value)) for key, value in images.items() if _is_image(value)]

    def _tile(self, frames: list[tuple[str, np.ndarray]]) -> np.ndarray | None:
        cv2 = self._cv2
        if not frames:
            return None
        target_h = 360
        tiles = []
        for name, img in frames:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif self.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            scale = target_h / h
            img = cv2.resize(img, (max(1, int(w * scale)), target_h))
            cv2.putText(img, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(img)
        strip = cv2.hconcat(tiles) if len(tiles) > 1 else tiles[0]
        if strip.shape[1] > self.max_width:
            scale = self.max_width / strip.shape[1]
            strip = cv2.resize(strip, (self.max_width, int(strip.shape[0] * scale)))
        return strip

    def _render(self, images: dict[str, Any] | None, status: dict[str, Any]) -> np.ndarray | None:
        cv2 = self._cv2
        strip = self._tile(self._select_images(images)) if images else None
        width = strip.shape[1] if strip is not None else max(640, self.max_width // 2)

        panel = np.zeros((self.panel_height, width, 3), dtype=np.uint8)
        self._draw_panel(panel, status)

        if strip is None:
            return panel
        return cv2.vconcat([strip, panel])

    def _draw_panel(self, panel: np.ndarray, status: dict[str, Any]) -> None:
        cv2 = self._cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (235, 235, 235)
        grey = (150, 150, 150)
        yellow = (0, 230, 230)
        green = (60, 230, 60)
        red = (60, 60, 235)

        def put(text, x, y, color=white, scale=0.5, thick=1):
            cv2.putText(panel, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        def put_fit(text, x, y, color, scale, thick, max_w):
            """Draw text, shrinking the font so it never overflows ``max_w`` pixels."""
            s = scale
            while s > 0.4:
                (tw, _th), _ = cv2.getTextSize(text, font, s, thick)
                if tw <= max_w:
                    break
                s -= 0.1
            cv2.putText(panel, text, (x, y), font, s, color, thick, cv2.LINE_AA)

        y = 24

        task = status.get("task")
        if task:
            put(f"TASK: {task}", 10, y, green, 0.6, 1)
            y += 26

        meta = []
        if status.get("step") is not None:
            meta.append(f"step {status['step']}")
        if status.get("fps") is not None:
            meta.append(f"{status['fps']:.1f} fps")
        if status.get("elapsed") is not None:
            elapsed = status["elapsed"]
            meta.append(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        if meta:
            put(" | ".join(meta), 10, y, grey, 0.5)
            y += 24

        if status.get("recording") is not None:
            rec = bool(status.get("recording"))
            ep = status.get("episode_index")
            buffered = status.get("buffered_frames")
            if rec:
                extra = ""
                if ep is not None and buffered is not None:
                    extra = f"   episode #{ep}  ({buffered} frames)"
                put(f"REC{extra}", 10, y, red, 0.6, 2)
            else:
                put("idle", 10, y, grey, 0.6, 2)
            back = status.get("back")
            if back is not None:
                put(
                    "BACK=1" if back else "back=0",
                    panel.shape[1] - 130,
                    y,
                    red if back else grey,
                    0.6,
                    2,
                )
            y += 26

            saved_eps = status.get("saved_episodes")
            saved_frames = status.get("saved_frames")
            if saved_eps is not None and saved_frames is not None:
                put(f"dataset: {saved_eps} episodes | {saved_frames} steps saved", 10, y, green, 0.5)
                y += 24

        # The subtask is drawn ~3x larger than everything else so the operator can read it
        # at a glance from across the workspace.
        skill_text = status.get("skill_text")
        if skill_text:
            put("SUBTASK:", 10, y, yellow, 0.6, 1)
            y += 36
            put_fit(skill_text, 10, y + 12, yellow, 1, 3, max_w=panel.shape[1] - 20)
            y += 48

        # Free-form `(text, color_name, scale)` rows from callers with more state to show,
        # e.g. the policy-rollout recorder (control source, teach button, queued handover).
        colors = {"white": white, "grey": grey, "yellow": yellow, "green": green, "red": red}
        for text, color, scale in status.get("lines") or ():
            # Text taller than the rows above needs extra room above its baseline.
            y += int(max(0.0, scale - 0.6) * 30)
            put_fit(text, 10, y, colors.get(color, white), scale, 2 if scale >= 0.8 else 1, panel.shape[1] - 20)
            y += max(22, int(40 * scale))
