from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from predict_rlm import RunTrace
from pydantic import BaseModel, Field

STATE_DIR = "fractal"
WORKSPACES_DIR = "workspaces"
SESSIONS_DIR = "sessions"
ENV_STATE_HOME = "FRACTAL_STATE_HOME"
SCHEMA_VERSION = 1
MAX_HISTORY_TURNS = 20
MAX_ITERATIONS_ERROR = (
    "Reached max iterations before SUBMIT; response came from fallback extraction."
)
INTERRUPTED_ERROR = "Turn interrupted by user."


def _new_session_id() -> str:
    return uuid.uuid4().hex


class UserTurn(BaseModel):
    message: str


class TurnUsage(BaseModel):
    """Host-recorded LM accounting for one agent turn.

    Derived from the PredictRLM RunTrace, not from model output, so it stays
    trustworthy across failed and interrupted turns.
    """

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)
    duration_ms: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    # Prompt tokens of the turn's final main-LM call. Fractal's RLM loop keeps
    # context bounded via summaries, so this is the live "context size" figure
    # rather than a cumulative count.
    context_tokens: int = Field(default=0, ge=0)


class AgentTurn(BaseModel):
    status: Literal["succeeded", "failed", "max_iterations", "interrupted"]
    response: str = ""
    files_read_count: int = Field(default=0, ge=0)
    files_changed_count: int = Field(default=0, ge=0)
    commands_run_count: int = Field(default=0, ge=0)
    error: str | None = None
    usage: TurnUsage | None = None


class SummaryTurn(BaseModel):
    turn_id: str
    user: UserTurn
    agent: AgentTurn | None = None


class SessionSummary(BaseModel):
    turns: list[SummaryTurn] = Field(default_factory=list)


class SessionHistoryTurn(BaseModel):
    turn_id: str
    user_message: str
    status: Literal["pending", "succeeded", "failed", "max_iterations", "interrupted"]
    files_read: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    trace: RunTrace | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class SessionState(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    session_id: str = Field(default_factory=_new_session_id)
    summary: SessionSummary = Field(default_factory=SessionSummary)
    history: list[SessionHistoryTurn] = Field(default_factory=list)


class FractalSession:
    """Workspace-scoped Fractal session state stored under global state."""

    def __init__(
        self, state: SessionState | None = None, *, session_id: str | None = None
    ) -> None:
        self.state = state or SessionState(session_id=session_id or _new_session_id())

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def summary_model(self) -> SessionSummary:
        return self.state.summary

    @property
    def history(self) -> list[SessionHistoryTurn]:
        return self.state.history

    @property
    def turns(self) -> list[SummaryTurn]:
        return self.state.summary.turns

    @classmethod
    def load(
        cls, workspace_path: str | Path, *, session_id: str | None = None
    ) -> "FractalSession":
        if session_id is None:
            # Multi-session storage exists before resume selection does. Until
            # the CLI has an explicit selector, each process gets a fresh ID
            # instead of silently choosing the wrong prior conversation.
            return cls()

        path = session_path(workspace_path, session_id)
        if not path.exists():
            return cls(session_id=session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            backup_path = _backup_bad_session(path)
            backup_note = (
                f"Preserved a backup at {backup_path}."
                if backup_path is not None
                else "Could not preserve a backup."
            )
            warnings.warn(
                f"Ignoring unreadable Fractal session at {path}: {exc}. "
                f"{backup_note}",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls(session_id=session_id)

        if not isinstance(data, dict):
            warnings.warn(
                f"Ignoring malformed Fractal session at {path}: expected a JSON object.",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls(session_id=session_id)

        if data.get("schema_version") == SCHEMA_VERSION:
            try:
                state = SessionState.model_validate(data)
            except ValueError as exc:
                warnings.warn(
                    f"Ignoring malformed Fractal session at {path}: {exc}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return cls(session_id=session_id)
            if state.session_id != session_id:
                warnings.warn(
                    f"Ignoring Fractal session at {path}: embedded session_id "
                    f"{state.session_id!r} does not match requested session_id "
                    f"{session_id!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return cls(session_id=session_id)
            return cls(state)

        warnings.warn(
            f"Ignoring unsupported Fractal session format at {path}: "
            f"expected schema_version={SCHEMA_VERSION}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cls(session_id=session_id)

    def save(self, workspace_path: str | Path) -> None:
        self._enforce_history_limit()
        path = session_path(workspace_path, self.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def summary(self) -> str:
        return render_session_summary(self.state.summary)

    def session_history_payload(self) -> list[dict[str, Any]]:
        # Full traces can be large. The structured summary is the durable
        # compressed trajectory, so history is bounded to exact-recall data.
        # Keep the RLM boundary JSON-safe: direct execution injects variables by
        # repr, so custom Pydantic model reprs would require host-only classes in
        # the runner globals.
        self._enforce_history_limit()
        return [turn.model_dump(mode="json") for turn in self.state.history]

    def add_user_message(self, content: str) -> str:
        turn_id = f"turn-{uuid.uuid4().hex}"
        now = _utc_now()
        self.state.summary.turns.append(
            SummaryTurn(turn_id=turn_id, user=UserTurn(message=content))
        )
        self.state.history.append(
            SessionHistoryTurn(
                turn_id=turn_id,
                user_message=content,
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
        self._enforce_history_limit()
        return turn_id

    def add_agent_turn(
        self,
        *,
        status: Literal["succeeded", "failed", "max_iterations", "interrupted"],
        response: str = "",
        changed_files: list[str] | None = None,
        files_read: list[str] | None = None,
        commands_run: list[str] | None = None,
        error: str | None = None,
        trace: RunTrace | None = None,
        turn_id: str | None = None,
    ) -> None:
        summary_turn = self._find_summary_turn(turn_id)
        if summary_turn is None:
            if turn_id is None:
                raise ValueError("Cannot add an agent turn before a user turn exists.")
            raise ValueError(f"No session summary turn found for id {turn_id!r}.")
        files_read_list = _require_string_list(files_read, "files_read")
        files_modified_list = _require_string_list(changed_files, "changed_files")
        commands_run_list = _require_string_list(commands_run, "commands_run")
        summary_turn.agent = AgentTurn(
            status=status,
            response=response,
            files_read_count=len(files_read_list),
            files_changed_count=len(files_modified_list),
            commands_run_count=len(commands_run_list),
            error=error,
            usage=turn_usage_from_trace(trace),
        )
        history_turn = self._find_history_turn(turn_id or summary_turn.turn_id)
        if history_turn is None:
            raise ValueError(
                f"No session history turn found for id {turn_id or summary_turn.turn_id!r}."
            )
        history_turn.status = status
        history_turn.files_read = files_read_list
        history_turn.files_modified = files_modified_list
        history_turn.commands_run = commands_run_list
        history_turn.trace = trace
        history_turn.error = error
        history_turn.updated_at = _utc_now()
        self._enforce_history_limit()

    def _find_summary_turn(self, turn_id: str | None) -> SummaryTurn | None:
        if turn_id is not None:
            for turn in reversed(self.state.summary.turns):
                if turn.turn_id == turn_id:
                    return turn
            return None
        return self.state.summary.turns[-1] if self.state.summary.turns else None

    def _find_history_turn(self, turn_id: str) -> SessionHistoryTurn | None:
        for turn in reversed(self.state.history):
            if turn.turn_id == turn_id:
                return turn
        return None

    def _enforce_history_limit(self) -> None:
        self.state.history = self.state.history[-MAX_HISTORY_TURNS:]

    def usage_totals(self) -> TurnUsage:
        return summarize_usage(self.state.summary)


def summarize_usage(summary: SessionSummary) -> TurnUsage:
    """Aggregate recorded usage across all turns in a session summary.

    ``context_tokens`` is not summed; it carries the most recent turn's live
    context size since the RLM loop re-summarizes between turns.
    """
    totals = TurnUsage()
    for turn in summary.turns:
        usage = turn.agent.usage if turn.agent is not None else None
        if usage is None:
            continue
        totals.input_tokens += usage.input_tokens
        totals.output_tokens += usage.output_tokens
        totals.cost += usage.cost
        totals.duration_ms += usage.duration_ms
        totals.iterations += usage.iterations
        if usage.context_tokens:
            totals.context_tokens = usage.context_tokens
    return totals


def turn_usage_from_trace(trace: RunTrace | None) -> TurnUsage | None:
    if trace is None:
        return None
    context_tokens = 0
    for step in reversed(trace.steps):
        prompt_tokens = step.usage.main.input_tokens
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            context_tokens = prompt_tokens
            break
    return TurnUsage(
        input_tokens=trace.usage.main.input_tokens + trace.usage.sub.input_tokens,
        output_tokens=trace.usage.main.output_tokens + trace.usage.sub.output_tokens,
        cost=trace.usage.main.cost + trace.usage.sub.cost,
        duration_ms=trace.duration_ms,
        iterations=trace.iterations,
        context_tokens=context_tokens,
    )


class HeadlessResult(BaseModel):
    """Machine-readable result of one non-interactive (`-p --json`) turn.

    Deliberately separate from ``SessionState``: this is the public CLI output
    contract, so it stays stable even when the on-disk session schema changes.
    It reports a single turn, not the cumulative session, and carries the real
    changed-file names from the live turn rather than the summary's counts.
    """

    session_id: str = ""
    workspace: str = ""
    status: Literal["succeeded", "failed", "max_iterations", "interrupted"]
    response: str = ""
    changed_files: list[str] = Field(default_factory=list)
    usage: TurnUsage | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"))


def headless_result_from_turn(
    *,
    session_id: str,
    workspace: str,
    response: str,
    changed_files: list[str],
    trace: RunTrace | None,
) -> HeadlessResult:
    """Build a headless result envelope from a completed turn's outputs."""
    status: Literal["succeeded", "max_iterations"] = "succeeded"
    if trace is not None and trace.status == "max_iterations":
        status = "max_iterations"
    return HeadlessResult(
        session_id=session_id,
        workspace=workspace,
        status=status,
        response=response,
        changed_files=list(changed_files),
        usage=turn_usage_from_trace(trace),
    )


def render_session_summary(summary: SessionSummary) -> str:
    if not summary.turns:
        return "No prior Fractal session context."

    # The summary is rendered into prompt text, but the source of truth stays
    # structured so future context builders can reason over the same artifact.
    lines = ["## Session Summary"]
    for index, turn in enumerate(summary.turns, start=1):
        lines.append("")
        lines.append(f"Turn {index} ({turn.turn_id})")
        lines.append(f"User: {turn.user.message}")
        if turn.agent is None:
            lines.append("Agent status: pending")
            continue
        lines.append(f"Agent status: {turn.agent.status}")
        if turn.agent.response:
            lines.append(f"Agent response: {turn.agent.response}")
        lines.append(f"files_read_count: {turn.agent.files_read_count}")
        lines.append(f"files_changed_count: {turn.agent.files_changed_count}")
        lines.append(f"commands_run_count: {turn.agent.commands_run_count}")
        if turn.agent.error:
            lines.append(f"Error: {turn.agent.error}")
    return "\n".join(lines)


class SessionInfo(BaseModel):
    """Lightweight listing entry for a stored workspace session."""

    session_id: str
    updated_at: str
    turn_count: int
    first_message: str


def list_sessions(workspace_path: str | Path) -> list[SessionInfo]:
    """List stored sessions for a workspace, most recently updated first.

    Unreadable or foreign files are skipped; listing is a navigation aid, not
    a validation pass.
    """
    sessions_dir = sessions_dir_path(workspace_path)
    if not sessions_dir.is_dir():
        return []
    entries: list[tuple[float, SessionInfo]] = []
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = SessionState.model_validate(data)
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        turns = state.summary.turns
        entries.append(
            (
                mtime,
                SessionInfo(
                    session_id=state.session_id,
                    updated_at=datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
                    turn_count=len(turns),
                    first_message=turns[0].user.message if turns else "",
                ),
            )
        )
    entries.sort(key=lambda item: item[0], reverse=True)
    return [info for _, info in entries]


def default_state_dir(
    *,
    env: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    environment = os.environ if env is None else env
    state_home = environment.get(ENV_STATE_HOME)
    if state_home:
        return Path(state_home).expanduser()
    xdg_state_home = environment.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / STATE_DIR
    home_path = Path.home() if home is None else Path(home).expanduser()
    return home_path / ".local" / "state" / STATE_DIR


def workspace_key(workspace_path: str | Path) -> str:
    resolved = Path(workspace_path).expanduser().resolve()
    slug_source = "-".join(part for part in resolved.parts if part != resolved.anchor)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug_source).strip("-._")
    if not slug:
        slug = "workspace"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    return f"{slug[:80]}--{digest}"


def workspace_state_dir(workspace_path: str | Path) -> Path:
    return default_state_dir() / WORKSPACES_DIR / workspace_key(workspace_path)


def sessions_dir_path(workspace_path: str | Path) -> Path:
    return workspace_state_dir(workspace_path) / SESSIONS_DIR


def session_path(workspace_path: str | Path, session_id: str) -> Path:
    _validate_session_id(session_id)
    return sessions_dir_path(workspace_path) / f"{session_id}.json"


def _validate_session_id(session_id: str) -> None:
    # Session IDs eventually become user-provided resume selectors. Keeping
    # them as plain filenames avoids turning resume support into path access.
    if not session_id or Path(session_id).name != session_id or session_id.endswith(
        ".json"
    ):
        raise ValueError(f"Invalid Fractal session id: {session_id!r}")


def _backup_bad_session(path: Path) -> Path | None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = path.with_suffix(f"{path.suffix}.bad-{stamp}")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_suffix(f"{path.suffix}.bad-{stamp}-{counter}")
        counter += 1
    try:
        shutil.copy2(path, backup_path)
    except OSError:
        return None
    return backup_path


def _require_string_list(value: list[str] | None, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be list[str] or None.")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
