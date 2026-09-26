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

# Ported from the LeRobot 0.6 working copy (lerobot.utils.record_controls); LeRobot 0.5.1 lacks it.
"""The interactive recorder's control state machine, independent of how it is driven.

One episode is started, paused, resumed, saved or discarded by the operator rather than
by a timer.  :class:`RecordControls` owns that state and nothing else: it holds no
hardware, writes no dataset, and knows nothing about keyboards or web sockets.  Two
front-ends drive the same instance through two entry points:

* :meth:`RecordControls.press` takes a canonical key name — the vocabulary
  :mod:`lerobot.utils.keyboard_input` emits, so ``pynput``, the terminal backend and
  :func:`~lerobot.utils.keyboard_input.press_key_remotely` all work unchanged — and
  resolves it through :data:`RECORD_KEYS`.
* :meth:`RecordControls.apply` takes an already-resolved action name, for a front-end
  that sends commands rather than keystrokes (the web worker's ``control`` commands).

The control loop reads plain :class:`threading.Event` flags (``recording``, ``save``,
``discard``, ``stop`` …); an optional ``on_change`` observer receives every transition so
a UI can mirror the state without polling.  Illegal transitions raise :class:`ValueError`
— the operator asked for something the state machine cannot do, and the caller decides
whether that is a log line (CLI) or a message back to the browser (web).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

#: The recording state machine's key bindings, mirroring the operator's existing ROS
#: collector so muscle memory carries over. The web UI renders one button per entry and
#: sends the key name; a physical keyboard produces the same names.
RECORD_KEYS: dict[str, str] = {
    "c": "start",  # begin, or resume a paused episode
    "space": "pause",  # pause/resume within the episode — does not split it
    "s": "save",  # save the buffered episode, back to idle
    "r": "discard",  # drop the buffer, back to idle
    "q": "stop",
    "esc": "stop",
}

#: Digit keys select the subtask index written into every frame.
MAX_SUBTASKS = 8

#: One-line legend, passed to `create_key_listener(controls_help=...)` and printed at startup.
RECORD_CONTROLS_HELP = "c=start/resume, space=pause, s=save, r=discard, 1-8=subtask, q=stop"


class RecordControls:
    """Start/pause/save/discard state for one interactive recording session.

    Args:
        subtasks: The subtask vocabulary. Digit keys ``1..len(subtasks)`` select the index
            written into every frame's ``subtask_index`` column. Empty disables subtasks.
        on_change: Called as ``on_change(event, payload)`` after every accepted transition.
            Events are ``"recording"``, ``"paused"``, ``"save"``, ``"discard"``,
            ``"subtask"`` and ``"stopping"``. Exceptions are logged and swallowed — a
            broken front-end must never wedge the state machine.
    """

    def __init__(
        self,
        subtasks: Sequence[str] = (),
        *,
        on_change: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.subtasks = list(subtasks)
        self._on_change = on_change
        self._lock = threading.Lock()

        # Flags the control loop polls once per tick.
        self.recording = threading.Event()
        self.stop = threading.Event()
        self.save = threading.Event()
        self.discard = threading.Event()
        self.emergency = threading.Event()

        self.subtask = 0
        #: What `stop` does with an episode still in the buffer: "save" or "discard".
        self.stop_pending = "discard"
        #: True once the operator has started at least once, so `pause` knows whether it
        #: is pausing a live episode or resuming a paused one.
        self.has_started = False

    # ------------------------------------------------------------------ entry points

    def press(self, key: str) -> None:
        """Apply a canonical key name (``"c"``, ``"space"``, ``"1"`` …).

        Digits within the subtask vocabulary select a subtask; everything else is looked
        up in :data:`RECORD_KEYS`. Unbound keys are ignored, because both local backends
        deliver *every* keystroke — arrow keys, letters typed by accident — and a session
        must not fall over because the operator brushed a key.
        """
        name = (key or "").lower()
        if name.isdigit() and 1 <= int(name) <= len(self.subtasks):
            self.apply("set_subtask", index=int(name) - 1)
            return
        action = RECORD_KEYS.get(name)
        if action is not None:
            self.apply(action)

    def apply(self, name: str, **args: Any) -> None:
        """Apply an already-resolved action name.

        Raises:
            ValueError: The action is not legal in the current state (pausing before the
                first start, changing subtask while recording, or an out-of-range subtask
                index).
        """
        with self._lock:
            if name in ("start", "resume"):
                self.has_started = True
                self.recording.set()
                self._emit("recording", {"message": "Recording"})
            elif name == "pause":
                self._pause_locked()
            elif name == "save":
                self.recording.clear()
                self.has_started = False
                self.save.set()
                self._emit("save", {"message": "Saving episode"})
            elif name == "discard":
                self.recording.clear()
                self.has_started = False
                self.discard.set()
                self._emit("discard", {"message": "Episode discarded"})
            elif name == "set_subtask":
                self._set_subtask_locked(int(args.get("index", 0)))
            elif name in ("stop", "emergency_stop"):
                self.stop_pending = str(args.get("pending", self.stop_pending))
                if name == "emergency_stop":
                    self.emergency.set()
                self._emit("stopping", {"message": "Stopping"})
                self.stop.set()

    # ------------------------------------------------------------------ transitions

    def _pause_locked(self) -> None:
        """Toggle recording within the *same* episode — pausing never splits it."""
        if self.recording.is_set():
            self.recording.clear()
            self._emit("paused", {"message": "Recording paused"})
        elif self.has_started:
            self.recording.set()
            self._emit("recording", {"message": "Recording"})
        else:
            raise ValueError("press start ('c') before pausing or resuming recording")

    def _set_subtask_locked(self, index: int) -> None:
        if self.recording.is_set():
            raise ValueError("pause recording before changing the subtask")
        if not 0 <= index < len(self.subtasks):
            raise ValueError(f"subtask index {index} is out of range")
        self.subtask = index
        self._emit(
            "subtask",
            {"subtask_index": index, "message": f"Subtask: {self.subtasks[index]}"},
        )

    # ------------------------------------------------------------------ observation

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(event, payload)
        except Exception:
            logger.exception("Error in RecordControls observer for event %r", event)

    @property
    def subtask_name(self) -> str:
        """The current subtask's name, or ``""`` when no vocabulary is configured."""
        return self.subtasks[self.subtask] if self.subtasks else ""

    def snapshot(self) -> dict[str, Any]:
        """A consistent read of the mutable state, for a status overlay or a log line."""
        with self._lock:
            return {
                "recording": self.recording.is_set(),
                "has_started": self.has_started,
                "subtask": self.subtask,
                "subtask_name": self.subtask_name,
                "stop_pending": self.stop_pending,
                "stopping": self.stop.is_set(),
            }
