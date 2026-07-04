from __future__ import annotations

from typing import Any

import dspy
from predict_rlm import Workspace

BASE_EDIT_WORKSPACE_INSTRUCTIONS = """Act as a focused coding agent over the mounted workspace.

You receive:
- `workspace`: mutable project workspace path.
- `included_paths`: optional additional mutable workspace paths for local
  files, directories, or project-adjacent resources.
- `user_message`: the user's current request.
- `session_history`: detailed prior Fractal turn history with file, command,
  and PredictRLM trace details.

Inspect and edit files primarily under the `workspace` path. Use
`included_paths` when the task needs other mounted host paths, and edit included
paths only when the user request requires it. Prefer pathlib/os operations rooted
at the mounted paths, and
prefer os.open with dir_fd/root_fd, os.pread/os.pwrite/os.ftruncate, and
temp-file plus os.replace patterns when they make edits safer.

Keep changes focused on the current user request. Inspect files before modifying
them, preserve unrelated content, and verify important edits. Return only a
concise user-facing response and a list of changed file paths, relative to the
primary workspace when possible.
"""


def build_edit_workspace_signature(
    rendered_session_summary: str,
    workspace_instructions: str = "",
) -> type[dspy.Signature]:
    """Build the per-turn Fractal coding-agent signature."""

    # Static workspace instructions come before the dynamic session summary so
    # the prompt keeps a stable cacheable prefix across turns.
    workspace_section = ""
    if workspace_instructions.strip():
        workspace_section = f"""
## Workspace instructions (AGENTS.md)

The workspace contains an AGENTS.md file with project-specific guidance from
the user. Follow it when working in this workspace; the current `user_message`
takes precedence if they conflict.

{workspace_instructions.strip()}
"""

    # The summary must be visible before the RLM chooses to inspect variables,
    # so it is baked into the instructions for this specific turn.
    summary = rendered_session_summary.strip() or "No prior Fractal session context."
    instructions = f"""{BASE_EDIT_WORKSPACE_INSTRUCTIONS}
{workspace_section}
## Always-visible session summary

{summary}

The summary above is compressed structured trajectory context and is always
visible. It preserves prior user messages and compressed agent results. For
exact prior REPL reasoning, code, outputs, tool calls, or predict calls, inspect
`session_history` from Python.
"""

    # The rendered session summary is intentionally embedded in the signature
    # instructions instead of declared as an InputField. PredictRLM currently
    # exposes InputFields primarily as REPL variables with prompt previews; for
    # always-visible memory we need prompt text. A future PredictRLM API may
    # support explicit prompt-only context fields separate from REPL variables.
    class EditWorkspaceWithSession(dspy.Signature):
        __doc__ = instructions

        workspace: Workspace = dspy.InputField(
            desc=(
                "Project workspace path. In direct execution mode this is a real "
                "host path that Python subprocesses can use."
            )
        )
        included_paths: list[Workspace] | None = dspy.InputField(
            desc=(
                "Additional mounted workspace paths. These are visible paths in "
                "direct execution mode."
            )
        )
        user_message: str = dspy.InputField(
            desc="The user's current request for this turn."
        )
        session_history: list[dict[str, Any]] = dspy.InputField(
            desc=(
                "Full prior Fractal turn history, including file, command, "
                "and PredictRLM trace details for exact recall from Python."
            )
        )

        response: str = dspy.OutputField(
            desc=(
                "Concise Markdown-formatted response to show in the CLI. Use "
                "Markdown for bullets, code spans, and short code blocks when helpful."
            )
        )
        changed_files: list[str] = dspy.OutputField(
            desc="Relative paths of files changed under the workspace path."
        )

    return EditWorkspaceWithSession


EditWorkspace = build_edit_workspace_signature("No prior Fractal session context.")
