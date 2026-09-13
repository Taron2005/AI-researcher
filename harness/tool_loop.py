"""
Generic multi-turn tool-calling loop, shared by any role that gets tools
(currently the Planner; the Software Engineer will reuse this too once it's
built). One place for the mechanics of "call the model, run any tool calls
it asks for, feed the results back, repeat until it gives a final answer"
-- so this isn't duplicated per role.

MAX_TOOL_TURNS bounds the session (CLAUDE.md's context-management
mitigation for Deep Thought failure #2: tool-calling reliability degrades
past ~20-50k tokens -- capping turns keeps a session well short of that).
"""

import json
from typing import Callable

from harness.llm_client import call_model
from harness.trace import Trace

MAX_TOOL_TURNS = 8


def run_tool_loop(
    model: str,
    temperature: float,
    messages: list[dict],
    tools: list[dict],
    dispatch: dict,
    online: bool = False,
    trace: Trace | None = None,
    stage: str = "unknown",
    max_turns: int = MAX_TOOL_TURNS,
    validate: Callable[[str], str | None] | None = None,
) -> str:
    """
    Advances `messages` forward until the model responds without requesting
    a tool call (and, if `validate` is given, that response passes
    validation), or `max_turns` is reached. Returns the model's final text
    content. Mutates `messages` in place (the caller doesn't need the full
    transcript back, only the final answer).

    `dispatch` maps a tool's function name to the plain Python function that
    actually performs it, e.g. {"search_papers": search_papers}.

    `max_turns` defaults to MAX_TOOL_TURNS but is overridable per call --
    the Planner's research turns and the Software Engineer's file-writing
    turns don't need the same cap (CLAUDE.md's context-management bound is
    about staying well under ~20-50k tokens per session, not one fixed
    number of turns for every role).

    `validate`, if given, is called on any no-tool-calls response; it
    returns None if the response is acceptable, or an error string
    describing what's wrong. On an error, that string is fed back to the
    model as a corrective message and the loop continues instead of
    returning -- this is Coscientist's own documented fix for exactly this
    failure (01-coscientist.md: "if the model emits [an invalid response],
    inject a message telling it to follow the format"), added after a real
    run hit a degenerate 2-token response instead of the required format
    (see DECISIONS.md).
    """
    for _ in range(max_turns):
        response = call_model(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            online=online,
            trace=trace,
            stage=stage,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            final_text = message.content or ""
            error = None
            if not final_text.strip():
                # A confirmed real failure mode, not a guess: some models
                # (verified directly against deepseek-v4-pro's raw API
                # response) put their actual output in a separate
                # `reasoning` field and leave `content` empty when they get
                # stuck mid-thought without committing to an action. Without
                # this check, that silently looked like a legitimate "I'm
                # done, nothing more to say" response. See DECISIONS.md.
                error = (
                    "Your response was empty and contained no tool calls. "
                    "You must either call a tool or provide your final "
                    "answer as text -- make sure to conclude with one or "
                    "the other, not stop mid-thought."
                )
            elif validate:
                error = validate(final_text)

            if error:
                messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "user", "content": error})
                if trace is not None:
                    trace.log_event(stage=stage, event_type="format_retry", error=error)
                continue
            return final_text

        # The assistant's tool-call request has to be replayed back into the
        # conversation before the tool results, or the next call is invalid.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            # A hallucinated/wrong tool name (e.g. `edit_file` when only
            # write_file/read_file/local_run exist) used to raise an
            # unguarded KeyError here, crashing the whole pipeline -- caught
            # for real live, not a hypothetical. Same "broken tool call
            # becomes text feedback, not a crashed harness" principle as the
            # try/except below, just extended to cover the lookup itself.
            if tool_call.function.name not in dispatch:
                result = (
                    f"Tool error: '{tool_call.function.name}' is not a real tool. "
                    f"Your actual available tools are: {sorted(dispatch.keys())}."
                )
            else:
                fn = dispatch[tool_call.function.name]
                try:
                    result = fn(**args)
                except Exception as e:
                    # A broken tool call becomes text feedback to the model, not
                    # a crashed harness -- same "runtime produces text, model
                    # reacts to it" pattern as Coscientist's traceback handling.
                    result = f"Tool error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
            if trace is not None:
                trace.log_event(
                    stage=stage,
                    event_type="tool_call",
                    tool=tool_call.function.name,
                    args=args,
                    # 2000 chars, not 300 -- the "run traces" deliverable is
                    # meant to be fully inspectable (CLAUDE.md rule 6); 300
                    # was losing real diagnostic substance from local_run/
                    # read_file/search_papers results with no other record
                    # of the full content anywhere (unlike write_file, whose
                    # full content is already in `args` above, and unlike a
                    # role's final answer, which ends up in a real file).
                    result_preview=str(result)[:2000],
                )

    raise RuntimeError(f"Tool loop exceeded max_turns={max_turns} without a final answer")
