from __future__ import annotations

import asyncio
import signal
import tomllib
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal TUI session models",
)
pytest.importorskip("prompt_toolkit", reason="prompt_toolkit is required for input")


class FakeRuntime:
    def __init__(
        self,
        tmp_path: Path,
        *,
        fail: bool = False,
        include_trace: bool = False,
        max_iterations: bool = False,
        interrupt: bool = False,
        interrupt_count: int = 1,
        emit_iteration_events: bool = False,
    ) -> None:
        from fractal.session import FractalSession

        self.workspace_path = tmp_path
        self.session = FractalSession(session_id="test-session")
        self.submitted: list[str] = []
        self.resumed: list[str] = []
        self.fail = fail
        self.include_trace = include_trace
        self.max_iterations = max_iterations
        self.interrupt_count = interrupt_count if interrupt else 0
        self.emit_iteration_events = emit_iteration_events
        self.provider_label = "openai-api"
        self.model_label = "gpt-5.5"
        self.sub_model_label = "gpt-5.5"
        self.applied_provider_selections: list[object] = []

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def new_session(self) -> None:
        from fractal.session import FractalSession

        self.session = FractalSession()

    def resume(self, session_id: str) -> None:
        from fractal.session import FractalSession, session_path

        self.resumed.append(session_id)
        if not session_path(self.workspace_path, session_id).exists():
            raise FileNotFoundError(f"No Fractal session found for id {session_id!r}.")
        self.session = FractalSession.load(self.workspace_path, session_id=session_id)

    def apply_provider_selection(
        self,
        selection: object,
        *,
        sub_model: str | None = None,
        sub_selection: object | None = None,
    ) -> None:
        if sub_model is None and sub_selection is not None:
            sub_model = sub_selection.model
        self.applied_provider_selections.append((selection, sub_model))
        self.provider_label = selection.provider
        self.model_label = selection.model
        self.sub_model_label = sub_model or selection.model

    async def submit(self, user_message: str, **kwargs: object) -> object:
        from fractal.agent.schema import FractalResult

        self.submitted.append(user_message)
        turn_id = self.session.add_user_message(user_message)
        on_pending = kwargs.get("on_pending")
        if on_pending is not None:
            pending_result = on_pending()
            if pending_result is not None:
                await pending_result
        interrupt_requested = kwargs.get("interrupt_requested")
        if interrupt_requested is not None and interrupt_requested():
            self.session.add_agent_turn(status="interrupted", turn_id=turn_id)
            raise asyncio.CancelledError
        if len(self.submitted) <= self.interrupt_count:
            signal.raise_signal(signal.SIGINT)
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                self.session.add_agent_turn(status="interrupted", turn_id=turn_id)
                raise
        if self.fail:
            self.session.add_agent_turn(
                status="failed",
                error="model failed",
                turn_id=turn_id,
            )
            raise RuntimeError("model failed")
        response = self._response(user_message)
        trace = self._trace() if self.include_trace or self.max_iterations else None
        if self.emit_iteration_events and trace is not None:
            from fractal.agent.schema import FractalIterationEvent
            from fractal.events import FractalRuntimeEvent

            on_runtime_event = kwargs.get("on_runtime_event")
            on_iteration_event = kwargs.get("on_iteration_event")
            if callable(on_runtime_event):
                on_runtime_event(
                    FractalRuntimeEvent(
                        kind="file_read",
                        target="builtins.open",
                        phase="before",
                        message="opening README.md",
                        path="README.md",
                    )
                )
            if callable(on_iteration_event):
                on_iteration_event(
                    FractalIterationEvent(
                        step=trace.steps[0],
                        max_iterations=trace.max_iterations,
                    )
                )
            if callable(on_runtime_event):
                on_runtime_event(
                    FractalRuntimeEvent(
                        kind="command",
                        target="subprocess.run",
                        phase="before",
                        message="running uv run pytest",
                        command="uv run pytest",
                    )
                )
            if callable(on_iteration_event):
                on_iteration_event(
                    FractalIterationEvent(
                        step=trace.steps[1],
                        max_iterations=trace.max_iterations,
                        is_final=True,
                    )
                )
        if self.max_iterations:
            from fractal.session import MAX_ITERATIONS_ERROR

            self.session.add_agent_turn(
                status="max_iterations",
                response=response,
                changed_files=[],
                trace=trace,
                turn_id=turn_id,
                error=MAX_ITERATIONS_ERROR,
            )
        else:
            self.session.add_agent_turn(
                status="succeeded",
                response=response,
                changed_files=[],
                trace=trace,
                turn_id=turn_id,
            )
        return FractalResult(
            response=response,
            trace=trace,
        )

    def _response(self, user_message: str) -> str:
        return f"response to {user_message}"

    def _trace(self) -> object:
        from predict_rlm.trace import IterationStep, RunTrace

        return RunTrace(
            status="max_iterations" if self.max_iterations else "completed",
            model="test-model",
            iterations=2,
            max_iterations=3,
            duration_ms=25,
            steps=[
                IterationStep(
                    iteration=1,
                    reasoning=(
                        "Inspect the files first and keep the reasoning long enough "
                        "to wrap in a narrow terminal."
                    ),
                    code="print('a')\nprint('b')",
                    output="hello",
                    untruncated_output="hello world",
                    duration_ms=10,
                ),
                IterationStep(
                    iteration=2,
                    reasoning="Apply the edit.",
                    code="x = 1",
                    output="done",
                    untruncated_output="done",
                    duration_ms=15,
                ),
            ],
        )


class FakePromptSession:
    def __init__(self, messages: list[str]) -> None:
        self.messages = messages
        self.prompts: list[object] = []
        self.prompt_kwargs: list[dict[str, object]] = []

    async def prompt_async(self, prompt: object, **kwargs: object) -> str:
        self.prompts.append(prompt)
        self.prompt_kwargs.append(kwargs)
        if not self.messages:
            raise EOFError
        return self.messages.pop(0)


def capture_console() -> tuple[Console, StringIO]:
    output = StringIO()
    return (
        Console(
            file=output,
            force_terminal=True,
            color_system=None,
            width=56,
            legacy_windows=False,
        ),
        output,
    )


def normalized_output(text: str) -> str:
    return " ".join(text.split())


def agent_statuses(runtime: FakeRuntime) -> list[str | None]:
    return [
        turn.agent.status if turn.agent is not None else None
        for turn in runtime.session.turns
    ]


def test_terminal_tui_renders_summary_as_native_output(tmp_path: Path) -> None:
    from rich.padding import Padding

    from fractal.tui.app import render_agent_response, render_summary, render_user_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("hello fractal")
    runtime.session.add_agent_turn(
        status="succeeded",
        response="hello human",
        changed_files=[],
        turn_id=turn_id,
    )
    console, output = capture_console()

    console.print(render_summary(runtime.session.summary_model))

    text = output.getvalue()
    assert "hello fractal" in text
    assert "hello human" in text
    user_message = render_user_message("hello fractal")
    assert user_message.renderables[1].spans[-1].style == "bold"
    agent_response = render_agent_response(runtime.session.turns[-1])
    assert isinstance(agent_response, Padding)
    assert agent_response.left == 2


def test_terminal_tui_prompt_label_is_purple() -> None:
    from fractal.tui.app import PROMPT_STYLE, render_prompt_label

    label = render_prompt_label()

    assert label.spans[0].style == "bold #8b5cf6"
    assert PROMPT_STYLE.style_rules[0] == ("prompt", "bold #8b5cf6")


def test_terminal_tui_prompt_continuation_has_no_ellipsis() -> None:
    from fractal.tui.app import _prompt_continuation

    assert _prompt_continuation(8, 1, True) == "        "
    assert "…" not in _prompt_continuation(8, 1, True)


def test_terminal_tui_prompt_renderer_matches_prompt_toolkit_height() -> None:
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.layout.controls import UIContent
    from prompt_toolkit.layout.screen import Screen, WritePosition
    from prompt_toolkit.output import DummyOutput

    from fractal.tui.app import PROMPT_STYLE, _FooterPromptSession, _prompt_continuation

    width = 24
    text = "alpha beta gamma delta epsilon"

    def line_prefix(lineno: int, wrap_count: int) -> object:
        if wrap_count == 0:
            return HTML("<prompt>fractal</prompt><session>›</session> ")
        return _prompt_continuation(width, lineno, True)

    with create_pipe_input() as pipe_input:
        session = _FooterPromptSession(
            style=PROMPT_STYLE,
            input=pipe_input,
            output=DummyOutput(),
            multiline=True,
            prompt_continuation=_prompt_continuation,
        )
        window = session._create_layout().current_window

    content = UIContent(
        get_line=lambda line_number: [("", text + " ")],
        line_count=1,
        cursor_position=Point(x=len(text), y=0),
    )
    estimated_height = content.get_height_for_line(0, width, line_prefix)
    screen = Screen()
    _, rowcol_to_yx = window._copy_body(
        content,
        screen,
        WritePosition(0, 0, width, 20),
        0,
        width,
        wrap_lines=True,
        get_line_prefix=line_prefix,
    )
    drawn_height = (
        max((position[0] for position in rowcol_to_yx.values()), default=0) + 1
    )

    assert drawn_height == estimated_height


def test_terminal_tui_footer_filler_uses_small_weight() -> None:
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.output import DummyOutput

    from fractal.tui.app import PROMPT_STYLE, _FooterPromptSession, _prompt_continuation

    with create_pipe_input() as pipe_input:
        session = _FooterPromptSession(
            style=PROMPT_STYLE,
            input=pipe_input,
            output=DummyOutput(),
            multiline=True,
            prompt_continuation=_prompt_continuation,
        )
        layout = session._create_layout()

    assert isinstance(layout.container, HSplit)
    filler = layout.container.children[-1]
    assert isinstance(filler, Window)
    assert filler.height.preferred == 0
    assert filler.height.weight == 1


def test_terminal_tui_renders_final_response_as_markdown(tmp_path: Path) -> None:
    from rich.markdown import Markdown

    from fractal.tui.app import FractalMarkdown, render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("format response")
    runtime.session.add_agent_turn(
        status="succeeded",
        response="**Done**\n\n- updated `README.md`",
        changed_files=[],
        turn_id=turn_id,
    )
    turn = runtime.session.turns[-1]

    message = render_agent_message(turn)

    assert isinstance(message, Markdown)
    assert isinstance(message, FractalMarkdown)


def test_terminal_tui_renders_max_iteration_response_as_incomplete(tmp_path: Path) -> None:
    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("finish task")
    from fractal.session import MAX_ITERATIONS_ERROR

    runtime.session.add_agent_turn(
        status="max_iterations",
        response="fallback response",
        changed_files=[],
        turn_id=turn_id,
        error=MAX_ITERATIONS_ERROR,
    )
    turn = runtime.session.turns[-1]

    panel = render_agent_message(turn)
    console, output = capture_console()
    console.print(panel)

    text = output.getvalue()
    assert "Reached max iterations; showing fallback response." in text
    assert "fallback response" in text


def test_terminal_tui_renders_interrupted_response(tmp_path: Path) -> None:
    from fractal.tui.app import render_agent_message

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("long task")
    runtime.session.add_agent_turn(status="interrupted", turn_id=turn_id)
    turn = runtime.session.turns[-1]

    rendered = render_agent_message(turn)

    assert "interrupted" in str(rendered)


def test_terminal_tui_renders_compact_trace_summary(tmp_path: Path) -> None:
    from fractal.tui.app import render_trace_summary

    runtime = FakeRuntime(tmp_path, include_trace=True)
    trace = runtime._trace()
    console, output = capture_console()

    console.print(render_trace_summary(trace))

    text = output.getvalue()
    normalized_text = normalized_output(text)
    assert "RLM turn 1/3" in text
    assert "reasoning: Inspect the files first" in text
    assert (
        "Inspect the files first and keep the reasoning long enough to wrap "
        "in a narrow terminal."
    ) in normalized_text
    assert "python: 2 lines" in text
    assert "output: 11 chars" in text
    assert "RLM turn 2/3" in text
    assert "python: 1 lines" in text
    assert "print('a')" not in text
    assert "code:" not in text


def test_terminal_tui_renders_verbose_trace_summary(tmp_path: Path) -> None:
    from fractal.tui.app import render_trace_summary

    runtime = FakeRuntime(tmp_path, include_trace=True)
    trace = runtime._trace()
    console, output = capture_console()

    console.print(render_trace_summary(trace, verbose=True))

    text = output.getvalue()
    assert "RLM turn 1/3" in text
    assert "reasoning: Inspect the files first" in text
    assert "python: 2 lines" in text
    assert "output: 11 chars" in text
    assert "code:" in text
    assert any(line.strip() == "code:" for line in text.splitlines())
    assert not any("code:" in line and "print" in line for line in text.splitlines())
    assert "print('a')" in text
    assert "print('b')" in text
    assert "output:" in text
    assert any(line.strip() == "output:" for line in text.splitlines())
    assert "hello" in text
    assert "hello world" not in text


def test_terminal_tui_renders_empty_verbose_trace_sections() -> None:
    from predict_rlm.trace import IterationStep

    from fractal.tui.app import render_trace_step

    step = IterationStep(
        iteration=1,
        reasoning="No-op.",
        code="",
        output="",
        untruncated_output="",
        duration_ms=1,
    )
    console, output = capture_console()

    console.print(render_trace_step(step, max_iterations=1, verbose=True))

    text = output.getvalue()
    assert "code:" in text
    assert "output:" in text
    assert text.count("(empty)") == 2


def test_terminal_tui_run_submits_and_prints_to_scrollback(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, include_trace=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("fix\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["fix"]
    assert "Running..." not in text
    assert "✓ complete" in text
    assert "RLM turn 1/3" in text
    assert "python: 2 lines" in text
    assert "output: 11 chars" in text
    assert "response to fix" in text


def test_terminal_tui_renders_live_iteration_events_once(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(
        tmp_path,
        include_trace=True,
        emit_iteration_events=True,
    )
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("fix\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert text.count("RLM turn 1/3") == 1
    assert text.count("RLM turn 2/3") == 1
    assert "reasoning: Inspect the files first" in text
    assert "reasoning: Apply the edit." in text
    assert (
        text.index("opening README.md")
        < text.index("RLM turn 1/3")
        < text.index("running uv run pytest")
        < text.index("RLM turn 2/3")
    )
    assert "response to fix" in text


def test_terminal_tui_renders_live_iteration_events_verbose(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(
        tmp_path,
        include_trace=True,
        emit_iteration_events=True,
    )
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("fix\n/exit\n"),
        verbose_iterations=True,
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert text.count("RLM turn 1/3") == 1
    assert text.count("code:") == 2
    assert "print('a')" in text
    assert "x = 1" in text
    assert "hello" in text
    assert "hello world" not in text
    assert (
        text.index("opening README.md")
        < text.index("RLM turn 1/3")
        < text.index("running uv run pytest")
        < text.index("RLM turn 2/3")
    )


def test_terminal_tui_failed_submit_renders_error_and_continues(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, fail=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("fail\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["fail"]
    assert "✗ failed" in text
    assert "model failed" in text
    assert "Running..." not in text


def test_terminal_tui_interrupted_submit_renders_and_continues(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, interrupt=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("stop\nnext\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["stop", "next"]
    assert "! turn interrupted" not in text
    assert "Turn interrupted by user." in text
    assert "response to next" in text
    assert "✓ complete" in text


def test_terminal_tui_handles_interrupt_before_submit_task_is_active(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO(""),
    )

    app._sigint_mode = "turn"
    app._handle_sigint(signal.SIGINT, None)
    result = asyncio.run(app.run_turn("early stop"))

    text = output.getvalue()
    assert result is None
    assert runtime.submitted == ["early stop"]
    assert runtime.session.turns[-1].agent is not None
    assert runtime.session.turns[-1].agent.status == "interrupted"
    assert "! turn interrupted" not in text
    assert "Turn interrupted by user." in text


def test_terminal_tui_run_turn_propagates_external_cancellation(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    class BlockingRuntime(FakeRuntime):
        async def submit(self, user_message: str, **kwargs: object) -> object:
            self.submitted.append(user_message)
            self.session.add_user_message(user_message)
            await asyncio.Event().wait()

    runtime = BlockingRuntime(tmp_path)
    console, _ = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO(""),
    )

    async def cancel_run_turn() -> None:
        task = asyncio.create_task(app.run_turn("shutdown"))
        for _ in range(10):
            if runtime.submitted:
                break
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_run_turn())

    assert runtime.submitted == ["shutdown"]
    assert runtime.session.turns[-1].agent is None
    assert runtime.session.history[-1].status == "pending"


def test_terminal_tui_sigint_updates_active_status_immediately(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp
    from fractal.tui.app import INTERRUPTING_STATUS

    class FakeStatus:
        def __init__(self) -> None:
            self.updates: list[str] = []

        def update(self, value: str) -> None:
            self.updates.append(value)

    runtime = FakeRuntime(tmp_path)
    app = TerminalFractalApp(
        runtime,
        input_stream=StringIO(""),
    )
    status = FakeStatus()

    app._sigint_mode = "turn"
    app._active_status = status
    app._handle_sigint(signal.SIGINT, None)

    assert app._turn_interrupt_requested is True
    assert status.updates == [INTERRUPTING_STATUS]


def test_terminal_tui_runtime_event_updates_active_status(tmp_path: Path) -> None:
    from fractal.events import FractalRuntimeEvent
    from fractal.tui import TerminalFractalApp
    from fractal.tui.app import render_runtime_event_log

    class FakeConsole:
        def __init__(self) -> None:
            self.prints: list[object] = []

        def print(self, value: object = "", **kwargs: object) -> None:
            self.prints.append(value)

    runtime = FakeRuntime(tmp_path)
    console = FakeConsole()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO(""),
    )

    app._show_runtime_event_status(
        FractalRuntimeEvent(
            kind="file_read",
            target="builtins.open",
            phase="before",
            message="opening [README].md",
            path="[README].md",
        )
    )

    assert [str(item) for item in console.prints] == [
        "  opening [README].md"
    ]
    assert console.prints[0].spans[0].style == "dim"
    assert console.prints[0].spans[1].style == "cyan"

    command = render_runtime_event_log(
        FractalRuntimeEvent(
            kind="command",
            target="subprocess.run",
            phase="before",
            message="running uv run pytest",
            command="uv run pytest",
        )
    )

    assert str(command) == "  running uv run pytest"
    assert command.spans[1].style == "magenta"


def test_terminal_tui_prompt_mode_sigint_does_not_escape_handler(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    app = TerminalFractalApp(
        runtime,
        input_stream=StringIO(""),
    )

    app._sigint_mode = "prompt"
    app._handle_sigint(signal.SIGINT, None)

    assert app._turn_interrupt_requested is False


def test_terminal_tui_two_interrupted_submits_do_not_exit(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, interrupt=True, interrupt_count=2)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("stop\nagain\nfinal\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["stop", "again", "final"]
    assert "! turn interrupted" not in text
    assert agent_statuses(runtime) == ["interrupted", "interrupted", "succeeded"]
    assert "response to final" in text
    assert "✓ complete" in text


def test_terminal_tui_late_prompt_sigint_after_interrupt_does_not_exit(
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    class LateSigintApp(TerminalFractalApp):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.messages = ["stop", "again", "final", "/exit"]
            self.injected_late_sigints = 0

        async def read_message(self) -> str | None:
            if len(runtime.submitted) in {1, 2} and self.injected_late_sigints < 2:
                self.injected_late_sigints += 1
                self._handle_sigint(signal.SIGINT, None)
            return self.messages.pop(0)

    runtime = FakeRuntime(tmp_path, interrupt=True, interrupt_count=2)
    console, output = capture_console()
    app = LateSigintApp(runtime, console=console)

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["stop", "again", "final"]
    assert app.injected_late_sigints == 2
    assert agent_statuses(runtime) == ["interrupted", "interrupted", "succeeded"]
    assert "response to final" in text


def test_terminal_tui_max_iterations_is_not_rendered_as_complete(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path, max_iterations=True)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("finish\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["finish"]
    assert "! max iterations" in text
    assert "✓ complete" not in text
    assert "Reached max iterations; showing fallback response." in text
    assert "response to finish" in text


def test_terminal_tui_uses_prompt_toolkit_session_for_live_input(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    prompt_session = FakePromptSession(["fix", "/exit"])
    app = TerminalFractalApp(
        runtime,
        console=console,
        prompt_session=prompt_session,
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == ["fix"]
    assert prompt_session.prompts
    assert prompt_session.prompt_kwargs[0]["wrap_lines"] is True
    assert prompt_session.prompt_kwargs[0]["set_exception_handler"] is False
    assert "response to fix" in text


def test_terminal_tui_resume_command_switches_sessions(tmp_path: Path) -> None:
    from fractal.session import FractalSession
    from fractal.tui import TerminalFractalApp

    existing = FractalSession(session_id="existing")
    turn_id = existing.add_user_message("prior request")
    existing.add_agent_turn(
        status="succeeded",
        response="prior response",
        changed_files=[],
        turn_id=turn_id,
    )
    existing.save(tmp_path)

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/resume existing\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.resumed == ["existing"]
    assert runtime.session_id == "existing"
    assert "resumed session existing" in text
    assert "prior request" in text
    assert "prior response" in text


def test_terminal_tui_resume_command_requires_session_id(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/resume\n/exit\n"),
    )

    asyncio.run(app.run())

    assert runtime.resumed == []
    assert "usage: /resume <session-id>" in output.getvalue()


def test_terminal_tui_resume_command_errors_for_missing_session(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/resume missing\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.resumed == ["missing"]
    assert runtime.session_id == "test-session"
    assert "No Fractal session found for id 'missing'." in text


def test_terminal_tui_provider_command_runs_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    input_stream = StringIO("/provider\n2\n1\n\n\n2\n\n/exit\n")
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=input_stream,
    )

    asyncio.run(app.run())

    config_path = tmp_path / "fractal" / "config.toml"
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert runtime.submitted == []
    assert len(runtime.applied_provider_selections) == 1
    assert data["active_provider"] == "openai-api"
    assert data["active_model"] == "gpt-5.5"
    assert data["providers"]["openai-api"] == {
        "auth_source": "env",
        "api_key_env": "OPENAI_API_KEY",
    }
    text = output.getvalue()
    assert "Fractal global config setup" in text
    assert "Fractal config written to" in text
    assert (
        "Provider updated for this session and saved as the default."
        in normalized_output(text)
    )
    assert "sk-secret-value" not in config_path.read_text(encoding="utf-8")


def test_terminal_tui_provider_command_rejects_arguments(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/provider setup\n/exit\n"),
    )

    asyncio.run(app.run())

    assert runtime.submitted == []
    assert "usage: /provider" in output.getvalue()


def test_terminal_tui_model_command_updates_current_provider_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    config_path = tmp_path / "fractal" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )
    runtime = FakeRuntime(tmp_path)
    console, _ = capture_console()
    input_stream = StringIO("/model\n2\n\n/exit\n")
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=input_stream,
    )

    asyncio.run(app.run())

    assert runtime.submitted == []
    assert runtime.model_label == "gpt-5.4"
    assert len(runtime.applied_provider_selections) == 1
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert data["active_provider"] == "openai-api"
    assert data["active_model"] == "gpt-5.4"


def test_terminal_tui_model_command_warns_when_project_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    config_path = tmp_path / "fractal" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )
    project_path = tmp_path / ".fractal" / "config.toml"
    project_path.parent.mkdir(parents=True)
    project_path.write_text(
        'schema_version = 1\nactive_model = "gpt-5.4-mini"\n',
        encoding="utf-8",
    )
    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/model\n2\n\n/exit\n"),
    )

    asyncio.run(app.run())

    text = normalized_output(output.getvalue())
    assert "overrides active_model" in text
    assert "project value wins on next launch" in text


def test_terminal_tui_provider_setup_uses_in_process_onboarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import FractalConfig, ProviderConfig
    from fractal.tui import TerminalFractalApp

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    calls: list[str] = []

    async def fake_async_prompt_for_config(**kwargs: object) -> FractalConfig:
        calls.append("async_prompt_for_config")
        return FractalConfig(
            active_provider="openai-api",
            active_model="gpt-5.5",
            providers={
                "openai-api": ProviderConfig(
                    auth_source="env",
                    api_key_env="OPENAI_API_KEY",
                )
            },
        )

    monkeypatch.setattr(
        "fractal.onboarding.async_prompt_for_config",
        fake_async_prompt_for_config,
    )
    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/provider\n/exit\n"),
    )

    asyncio.run(app.run())

    assert calls == ["async_prompt_for_config"]
    assert runtime.submitted == []
    assert len(runtime.applied_provider_selections) == 1
    assert (
        "Provider updated for this session and saved as the default."
        in normalized_output(output.getvalue())
    )


def test_terminal_tui_model_setup_uses_model_only_onboarding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    config_path = tmp_path / "fractal" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )
    calls: list[str] = []

    async def fake_async_prompt_for_model(**kwargs: object) -> str:
        calls.append(kwargs["provider"].id)
        return "gpt-5.4-mini"

    async def fake_async_prompt_for_sub_model(**kwargs: object) -> str | None:
        calls.append(f"sub:{kwargs['main_model']}")
        return None

    monkeypatch.setattr(
        "fractal.onboarding.async_prompt_for_model",
        fake_async_prompt_for_model,
    )
    monkeypatch.setattr(
        "fractal.onboarding.async_prompt_for_sub_model",
        fake_async_prompt_for_sub_model,
    )
    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/model\n/exit\n"),
    )

    asyncio.run(app.run())

    assert calls == ["openai-api", "sub:gpt-5.4-mini"]
    assert runtime.submitted == []
    assert runtime.model_label == "gpt-5.4-mini"
    assert "Model updated to gpt-5.4-mini" in normalized_output(
        output.getvalue()
    )


def test_terminal_tui_verbose_command_toggles_session_verbosity(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/verbose\n/verbose off\n/verbose nope\n/exit\n"),
    )

    asyncio.run(app.run())

    assert runtime.submitted == []
    text = output.getvalue()
    assert "verbose iteration output on" in text
    assert "verbose iteration output off" in text
    assert "fractal config set defaults.verbose" in normalized_output(text)
    assert "usage: /verbose [on|off]" in text


def test_slash_command_completer_lists_commands() -> None:
    from prompt_toolkit.document import Document

    from fractal.tui.app import SlashCommandCompleter

    completer = SlashCommandCompleter()

    model = list(completer.get_completions(Document("/m"), None))
    provider = list(completer.get_completions(Document("/p"), None))
    resume = list(completer.get_completions(Document("/r"), None))
    verbose = list(completer.get_completions(Document("/v"), None))
    none_after_space = list(completer.get_completions(Document("/resume "), None))

    assert [completion.text for completion in model] == ["/model"]
    assert [completion.text for completion in provider] == ["/provider"]
    assert [completion.text for completion in resume] == ["/resume"]
    assert [completion.text for completion in verbose] == ["/verbose"]
    assert none_after_space == []


def test_terminal_tui_model_command_also_chooses_sub_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    config_path = tmp_path / "fractal" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )
    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/model\n1\n4\n/exit\n"),
    )

    asyncio.run(app.run())

    assert runtime.submitted == []
    assert len(runtime.applied_provider_selections) == 1
    selection, sub_model = runtime.applied_provider_selections[0]
    assert selection.model == "gpt-5.5"
    assert sub_model == "gpt-5.4-mini"
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert data["active_model"] == "gpt-5.5"
    assert data["active_sub_model"] == "gpt-5.4-mini"
    assert "sub gpt-5.4-mini" in normalized_output(output.getvalue())


def _usage_trace() -> object:
    from predict_rlm.trace import (
        IterationStep,
        LMUsage,
        RunTrace,
        TokenUsage,
    )

    return RunTrace(
        status="completed",
        model="test-model",
        iterations=2,
        max_iterations=3,
        duration_ms=12_400,
        usage=LMUsage(
            main=TokenUsage(input_tokens=8200, output_tokens=400, cost=0.0413),
        ),
        steps=[
            IterationStep(
                iteration=1,
                reasoning="r",
                code="x = 1",
                output="ok",
                untruncated_output="ok",
                duration_ms=10,
                usage=LMUsage(
                    main=TokenUsage(input_tokens=7421, output_tokens=141)
                ),
            ),
        ],
    )


def test_turn_footer_includes_usage_and_context() -> None:
    from fractal.agent.schema import FractalResult
    from fractal.tui.app import render_turn_footer

    footer = render_turn_footer(
        FractalResult(response="done", trace=_usage_trace())
    ).plain

    assert "✓ complete" in footer
    assert "2 iterations" in footer
    assert "12.4s" in footer
    assert "8.2k in / 400 out" in footer
    assert "7.4k ctx" in footer
    assert "$0.0413" not in footer


def test_turn_footer_without_trace_is_plain_complete() -> None:
    from fractal.agent.schema import FractalResult
    from fractal.tui.app import render_turn_footer

    footer = render_turn_footer(FractalResult(response="done")).plain

    assert footer == "✓ complete"


def test_bottom_toolbar_shows_context_estimate_not_accumulated_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.session import TurnUsage
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("fix")
    runtime.session.add_agent_turn(status="succeeded", response="ok", turn_id=turn_id)
    last_turn = runtime.session.turns[-1]
    assert last_turn.agent is not None
    last_turn.agent.usage = TurnUsage(
        input_tokens=8200,
        output_tokens=400,
        cost=0.0413,
    )
    monkeypatch.setattr(
        "fractal.tui.app.estimate_next_context_tokens",
        lambda runtime: 7421,
    )
    app = TerminalFractalApp(runtime)

    toolbar = "".join(fragment for _, fragment in app._bottom_toolbar_fragments())

    assert "~7.4k ctx" in toolbar
    assert "8.6k tok" not in toolbar
    assert "$0.04" not in toolbar


def test_bottom_toolbar_omits_context_when_estimate_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    def fail(runtime: object) -> int:
        raise RuntimeError("cannot estimate")

    monkeypatch.setattr("fractal.tui.app.estimate_next_context_tokens", fail)
    runtime = FakeRuntime(tmp_path)
    app = TerminalFractalApp(runtime)

    toolbar = "".join(fragment for _, fragment in app._bottom_toolbar_fragments())

    assert "model gpt-5.5" in toolbar
    assert "verbose off" in toolbar
    assert "ctx" not in toolbar


def test_bottom_toolbar_caches_context_estimate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.tui import TerminalFractalApp

    calls = 0

    def estimate(runtime: object) -> int:
        nonlocal calls
        calls += 1
        return 7421

    monkeypatch.setattr("fractal.tui.app.estimate_next_context_tokens", estimate)
    runtime = FakeRuntime(tmp_path)
    app = TerminalFractalApp(runtime)

    app._bottom_toolbar_fragments()
    app._bottom_toolbar_fragments()

    assert calls == 1


def test_terminal_tui_help_command_lists_commands(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/help\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == []
    assert "/resume" in text
    assert "/sessions" in text
    assert "/usage" in text


def test_terminal_tui_unknown_command_is_not_submitted(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/bogus\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == []
    assert "unknown command: /bogus" in text


def test_terminal_tui_path_prompt_is_submitted(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, _ = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/tmp/thing.py explain this\n/exit\n"),
    )

    asyncio.run(app.run())

    assert runtime.submitted == ["/tmp/thing.py explain this"]


def test_terminal_tui_new_command_starts_fresh_session(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    old_session_id = runtime.session_id
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/new\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert runtime.submitted == []
    assert runtime.session_id != old_session_id
    assert "started new session" in text


def test_terminal_tui_usage_command_reports_totals(tmp_path: Path) -> None:
    from fractal.session import TurnUsage
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    turn_id = runtime.session.add_user_message("fix")
    runtime.session.add_agent_turn(status="succeeded", response="ok", turn_id=turn_id)
    last_turn = runtime.session.turns[-1]
    assert last_turn.agent is not None
    last_turn.agent.usage = TurnUsage(
        input_tokens=8200,
        output_tokens=400,
        cost=0.0413,
        duration_ms=12_400,
        iterations=2,
        context_tokens=7421,
    )
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/usage\n/exit\n"),
    )

    asyncio.run(app.run())

    text = normalized_output(output.getvalue())
    assert "input tokens 8,200" in text
    assert "output tokens 400" in text
    assert "~7,421 tokens" in text
    assert "$0.0413" in text


def test_terminal_tui_sessions_command_lists_stored_sessions(tmp_path: Path) -> None:
    from fractal.session import FractalSession
    from fractal.tui import TerminalFractalApp

    stored = FractalSession()
    stored.add_user_message("add a login page")
    stored.save(tmp_path)
    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/sessions\n/exit\n"),
    )

    asyncio.run(app.run())

    text = normalized_output(output.getvalue())
    assert stored.session_id in text
    assert "add a login page" in text


def test_terminal_tui_verbose_command_toggles(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(
        runtime,
        console=console,
        input_stream=StringIO("/verbose\n/verbose\n/exit\n"),
    )

    asyncio.run(app.run())

    text = output.getvalue()
    assert "verbose iteration output on" in text
    assert "verbose iteration output off" in text


def test_terminal_tui_header_always_shows_sub_model(tmp_path: Path) -> None:
    from fractal.tui import TerminalFractalApp

    runtime = FakeRuntime(tmp_path)
    console, output = capture_console()
    app = TerminalFractalApp(runtime, console=console, input_stream=StringIO("/exit\n"))

    asyncio.run(app.run())

    text = normalized_output(output.getvalue())
    assert "model gpt-5.5 | sub gpt-5.5" in text

    runtime.sub_model_label = "gpt-5.4-mini"
    console, output = capture_console()
    app = TerminalFractalApp(runtime, console=console, input_stream=StringIO("/exit\n"))

    asyncio.run(app.run())

    assert "model gpt-5.5 | sub gpt-5.4-mini" in normalized_output(output.getvalue())
