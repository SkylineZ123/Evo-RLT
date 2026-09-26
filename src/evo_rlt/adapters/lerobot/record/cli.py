from __future__ import annotations

import argparse
import sys

from evo_rlt.adapters.lerobot.record.common import RESUME_LATEST
from evo_rlt.adapters.lerobot.record.runner import run_collect, run_full, run_live, run_segment


def run_teleop(args: argparse.Namespace) -> None:
    from evo_rlt.adapters.lerobot.record.teleop_collect import run_teleop as _run_teleop

    _run_teleop(args)


def run_replay(args: argparse.Namespace) -> None:
    from evo_rlt.adapters.lerobot.record.replay import run_replay as _run_replay

    _run_replay(args)


DEFAULT_COLLECT_DATASET_TAG = "vla_rlt_vla_test"
DEFAULT_COLLECT_TASK = "Insert the copper screw into the black sleeve."


def add_resume_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_LATEST,
        default=None,
        metavar="DATASET_ROOT",
        help=(
            "Append to an existing dataset instead of creating a new one. Without a value (YAML "
            "`resume: true`), continue the most recent dataset of --dataset-tag that has saved "
            "episodes. --num-episodes then counts this session's episodes."
        ),
    )


def add_common_record_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-source", choices=["vla", "teleop"], required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--reset-time-s", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=None)
    add_resume_arg(parser)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--default-episode-success", choices=["success", "failure"], default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", default=False)


def add_rtc_args(
    parser: argparse.ArgumentParser,
    *,
    execution_horizon_default: int = 10,
    vla_execution_horizon_default: int | None = None,
    action_queue_default: int | None = None,
) -> None:
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=execution_horizon_default)
    parser.add_argument("--vla-rtc-execution-horizon", type=int, default=vla_execution_horizon_default)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        default="EXP",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
    )
    parser.add_argument(
        "--rtc-action-queue-size-to-get-new-actions",
        type=int,
        default=action_queue_default,
    )


def add_default_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument("--task", default=DEFAULT_COLLECT_TASK)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_COLLECT_DATASET_TAG)
    add_resume_arg(parser)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--double-tap-window-s", type=float, default=0.6)
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rlt-toggle-key", default="r")
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--default-episode-success", choices=["success", "failure"], default=None)
    parser.add_argument(
        "--start-with-teleop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start each episode in teleop instead of VLA.",
    )
    parser.add_argument(
        "--only-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record only the RLT critical segment. The first RLT key press starts "
            "recording and enters RLT; the next RLT key press saves the segment. "
            "The default records the full trajectory immediately and uses the RLT key "
            "as the full-episode outcome key."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified real-robot recording entrypoint",
        epilog=(
            "Any subcommand also takes --config PATH.yaml: its keys are the option names "
            "(e.g. policy_path), an optional `command:` key selects the subcommand, and "
            "flags given on the command line override the file. See configs/record/."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="Run the default VLA-RLT-VLA real-robot data collection script.",
    )
    add_default_collect_args(collect)
    add_rtc_args(
        collect,
        vla_execution_horizon_default=25,
        action_queue_default=30,
    )
    collect.set_defaults(func=run_collect)

    segment = subparsers.add_parser(
        "segment",
        help="Record only the key segment. Success/failure labels apply to the segment.",
    )
    add_common_record_args(segment)
    add_rtc_args(segment)
    segment.add_argument("--critical-source", choices=["rlt", "vla"], required=True)
    segment.add_argument("--double-tap-window-s", type=float, default=0.6)
    segment.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    segment.add_argument("--chunk-exec-steps", type=int, default=25)
    segment.add_argument("--intervention-action-blend-time-s", type=float, default=0.4)
    segment.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    segment.set_defaults(func=run_segment)

    full = subparsers.add_parser(
        "full",
        help="Record the full trajectory. Success/failure labels apply to the trajectory.",
    )
    add_common_record_args(full)
    add_rtc_args(full)
    full.add_argument("--phase-mode", default=None)
    full.add_argument("--chunk-exec-steps", type=int, default=None)
    full.add_argument("--pedal-outcome", action=argparse.BooleanOptionalAction, default=False)
    full.add_argument("--episode-outcome-key", default="r")
    full.add_argument("--double-tap-window-s", type=float, default=0.6)
    full.set_defaults(func=run_full)

    teleop = subparsers.add_parser(
        "teleop",
        help="Record Piper teleoperation demos interactively (c=start, space=pause, s=save, r=discard).",
    )
    teleop.add_argument("--setup-json", default=None)
    teleop.add_argument("--task", required=True)
    teleop.add_argument("--num-episodes", type=int, default=50, help="Session ends after this many saves.")
    teleop.add_argument("--fps", type=int, default=30)
    teleop.add_argument("--dataset-tag", default="piper_teleop")
    add_resume_arg(teleop)
    teleop.add_argument("--vcodec", default="h264")
    teleop.add_argument("--align-time-s", type=float, default=3.0)
    teleop.add_argument("--stop-pending", choices=["save", "discard"], default="discard",
                        help="What q does with an unsaved episode still in the buffer.")
    teleop.add_argument("--status-view", action=argparse.BooleanOptionalAction, default=False)
    teleop.add_argument("--log-level", default="INFO")
    teleop.add_argument("--dry-run", action="store_true", default=False)
    teleop.set_defaults(func=run_teleop)

    replay = subparsers.add_parser(
        "replay",
        help="Replay recorded Piper episodes on the follower and report tracking error (dataset QA).",
    )
    replay.add_argument("--setup-json", default=None)
    replay.add_argument("--dataset-root", required=True,
                        help="Local dataset dir, e.g. <datasets.root>/0924_piper_teleop/teleop_190416.")
    replay.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode indices (default: all).")
    replay.add_argument("--source", choices=["action", "state"], default="action",
                        help="Stream the recorded action, or the recorded follower state.")
    replay.add_argument("--speed", type=float, default=1.0, help="Playback speed factor in (0, 1].")
    replay.add_argument("--approach-time-s", type=float, default=3.0,
                        help="Minimum time to ramp onto each episode's first frame.")
    replay.add_argument("--max-step-deg", type=float, default=10.0,
                        help="Skip episodes with a larger joint jump between consecutive frames.")
    replay.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=True,
                        help="Wait for Enter before each episode so the scene can be reset.")
    replay.add_argument("--save-trace", default=None,
                        help="Directory for per-episode npz files (command, recorded/measured state).")
    replay.add_argument("--log-level", default="INFO")
    replay.add_argument("--dry-run", action="store_true", default=False)
    replay.set_defaults(func=run_replay)

    live = subparsers.add_parser("live", help="Run policy live on the robot without saving a dataset.")
    live.add_argument("--policy-path", required=True)
    live.add_argument("--eval-script", required=True)
    live.add_argument("--vla-path", default=None)
    live.add_argument("--rl-token-path", default=None)
    live.add_argument("--phase-mode", default="always_vla")
    live.add_argument("--chunk-exec-steps", type=int, default=25)
    live.add_argument("--task", default="Insert the copper screw into the black sleeve.")
    live.add_argument("--duration", type=float, default=120.0)
    live.add_argument("--fps", type=float, default=30.0)
    live.add_argument("--setup-json", default=None)
    live.add_argument("--dry-run", action="store_true", default=False)
    add_rtc_args(live)
    live.set_defaults(func=run_live)
    return parser


CONFIG_COMMAND_KEY = "command"


def _pop_config_path(argv: list[str]) -> tuple[str | None, list[str]]:
    """Strip `--config PATH` / `--config=PATH` from argv, wherever it appears."""
    config_path = None
    rest: list[str] = []
    it = iter(argv)
    for token in it:
        if token == "--config":
            config_path = next(it, None)
            if config_path is None:
                raise SystemExit("--config needs a YAML file path")
        elif token.startswith("--config="):
            config_path = token.split("=", 1)[1]
        else:
            rest.append(token)
    return config_path, rest


def _subcommand_parsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise RuntimeError("record parser has no subcommands")


def _coerce_config_value(action: argparse.Action, key: str, value):
    if value is None:
        return None
    is_flag = isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction))
    if is_flag:
        if not isinstance(value, bool):
            raise SystemExit(f"config key '{key}' must be true or false, got {value!r}")
        return value
    if action.nargs == "?" and action.const is not None and isinstance(value, bool):
        # `key: true` is the bare flag (`--resume`); `key: false` leaves it out.
        return action.const if value else action.default
    if isinstance(value, bool):
        raise SystemExit(f"config key '{key}' does not take true/false, got {value!r}")
    if action.type in (int, float):
        try:
            value = action.type(value)
        except (TypeError, ValueError):
            raise SystemExit(f"config key '{key}' must be {action.type.__name__}, got {value!r}") from None
    elif not isinstance(value, str):
        value = str(value)
    if action.choices is not None and value not in action.choices:
        raise SystemExit(f"config key '{key}' must be one of {list(action.choices)}, got {value!r}")
    return value


def apply_config_file(parser: argparse.ArgumentParser, config_path: str, argv: list[str]) -> list[str]:
    """Load a YAML record config as subcommand defaults; explicit CLI flags still win.

    YAML keys are the CLI option names with `_` or `-` (`policy_path` == `--policy-path`).
    An optional top-level `command` picks the subcommand when argv does not name one.
    """
    import yaml

    with open(config_path) as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path}: top level must be a mapping of option -> value")

    subparsers = _subcommand_parsers(parser)
    # The top-level parser has no options of its own, so a subcommand can only be argv[0].
    cli_command = argv[0] if argv and argv[0] in subparsers else None
    config_command = config.pop(CONFIG_COMMAND_KEY, None)
    if cli_command is None:
        if config_command is None:
            raise SystemExit(f"{config_path}: set `command:` or pass the subcommand on the command line")
        if config_command not in subparsers:
            raise SystemExit(f"{config_path}: unknown command {config_command!r}; expected one of {list(subparsers)}")
        argv = [config_command, *argv]
    elif config_command is not None and config_command != cli_command:
        raise SystemExit(f"{config_path} is a '{config_command}' config, but '{cli_command}' was requested")
    command = cli_command or config_command

    subparser = subparsers[command]
    actions = {action.dest: action for action in subparser._actions if action.dest != "help"}
    defaults = {}
    for raw_key, value in config.items():
        key = str(raw_key).replace("-", "_")
        action = actions.get(key)
        if action is None:
            raise SystemExit(
                f"{config_path}: unknown option '{raw_key}' for '{command}'. Valid options: {sorted(actions)}"
            )
        if value is None:
            # `key: null` leaves the built-in default (or the CLI) in charge.
            continue
        defaults[key] = _coerce_config_value(action, raw_key, value)
        # Supplied by the config, so the CLI no longer has to repeat it.
        action.required = False
    subparser.set_defaults(**defaults)
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    config_path, argv = _pop_config_path(argv)
    if config_path is not None:
        argv = apply_config_file(parser, config_path, argv)
    args = parser.parse_args(argv)
    if args.command in {"segment", "full"} and args.dataset_tag is None:
        args.dataset_tag = f"{args.initial_source}_{args.command}"
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.func(args)


def collect_default_main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    main(["collect", *args])


if __name__ == "__main__":
    main()
