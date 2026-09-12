# 01 — Coscientist

- **Published:** Boiko, MacKnight, Kline, Gomes. *Autonomous chemical research with large language models.* Nature 624, 570–578 (2023). https://doi.org/10.1038/s41586-023-06792-0
- **Preprint (more appendices / the quotes below):** Boiko, MacKnight, Gomes. *Emergent autonomous scientific research capabilities of large language models.* arXiv:2304.05332 (Apr 2023).
- **Code (toy harness, not the full lab system):** https://github.com/gomesgroup/coscientist

Two versions of the same system. The Nature paper is the cleaned-up story (six tasks, ECL HPLC actually ran, yield-optimization game). The preprint has the rawer agent-loop details you quoted (ten-step budget, ada / 7800 tokens, SymPy traceback). Quotes below mix both.

They did **not** release the real Planner prompts. Safety. The GitHub repo is a stripped-down command loop so you can see the message format.

---

## What this system is

One GPT-4 chat (the **Planner**) with four text commands. Not a fleet of specialist agents arguing. Two of the commands call other LLMs. Two do not.

Plain-text user prompt in, e.g. `"perform multiple Suzuki reactions"`. The Planner is told to reason, search, calculate quantities, then run the reaction. It is also told it typically needs **at least ten steps**, and **not to ask the user clarifying questions** if the prompt is detailed enough. In the public toy code, that ten is literally `max_steps=10`.

---

## Chat roles (this is the loop)

GPT-4 chat completions have three roles. They use them like this:

| Role | What goes in it |
|---|---|
| **system** | Static Planner prompt: who you are, the four commands, output format (“reason, then one command”). |
| **user** | The scientist’s original prompt **and** every tool return (search snippets, Python stdout/stderr, docs, hardware errors). |
| **assistant** | Planner thoughts + exactly one command line. |

So tool output is not a special “function” role (this is pre-OpenAI-tools). It is just another user message. The next assistant turn sees the whole history.

Public toy code does the same: parse one `COMMAND <arg>` line, run the tool, append the result as `role: user`, loop.

Format they enforce: **one command per turn**. If the model emits zero commands or two, they inject a user message telling it to follow the format.

---

## The four commands (Planner action space)

Defined in the system prompt.

| Command | Goes to | Uses an LLM? | What it actually does |
|---|---|---|---|
| `GOOGLE` | Web searcher | Yes (GPT-4 or GPT-3.5) | Rewrite query → Google Search API → drop PDFs, keep first ~10 pages → `BROWSE` pages → summarize back to Planner. They say GPT-3.5 is fine here (faster, quality close). |
| `PYTHON` | Code execution | **No** | Run the Planner’s Python in a sandbox. stdout / traceback come back as a user message. Used for stoichiometry, volumes, later RDKit. |
| `DOCUMENTATION` | Docs searcher | Yes | Retrieve + summarize hardware API docs so the Planner can write valid `EXPERIMENT` code. |
| `EXPERIMENT` | Automation | **No** | Run generated protocol on hardware, or dump a manual procedure. In this paper: Opentrons Python API and ECL SLL. |

`GOOGLE` and `DOCUMENTATION` are the same pattern: Planner asks a question in English, a retrieval+LLM module answers in English, Planner continues.

---

## What they actually showed (Nature, six tasks)

1. Plan syntheses of known compounds from the public web (ibuprofen etc.). Search-GPT-4 beats GPT-4 with no search; non-search models hallucinate ibuprofen.
2. Navigate hardware docs (OT-2 API, including heater–shaker, which was **after** GPT-4’s training cutoff).
3. Use those docs to emit ECL SLL and run a real cloud-lab HPLC (caffeine standard).
4. Low-level OT-2 control (“draw a red cross on a 96-well plate”).
5. Multi-module: liquid handler + UV-Vis to identify three unknown dye wells; then Suzuki + Sonogashira on the OT-2 with GC-MS confirmation.
6. Closed-loop yield “game” on lookup-table Suzuki and Buchwald–Hartwig datasets (JSON condition picks, 20 iterations).

Suzuki/Sonogashira run: plates were moved by a human; **no human chose reagents or wrote the protocol**. Wrong heater–shaker method name → `DOCUMENTATION` → corrected protocol → products seen by GC-MS.

---

## Docker / code execution — what they mean, why it is interesting

The Planner is an LLM. It can emit `import os; os.system(...)`. If you `exec()` that on your laptop, the model has your files, network, and GPU.

Their move: **the code runner is not an LLM**. It is a dumb interpreter inside an **isolated Docker container**. The container has a small allowed package set. The host is not the same filesystem. stdout and the traceback are the only things that leave.

“Unexpected actions by the Planner” = the model wrote dangerous or just broken code. Isolation is for both.

This is a **trust boundary**, not a chemistry trick:

```
Planner (untrusted text) → PYTHON command → Docker Python → text result → back into the chat as a user message
```

The Planner then *sees* the failure and can rewrite the code. That second part is the self-correction story below.

**Note:** the public `simple_implementation` does **not** use Docker. It is `eval()` on a calculator toy. Docker is in the paper’s real runner, not in the released demo.

**To discuss later:** what the container is allowed to import (SymPy was *not* installed — that is why the traceback happened), whether `EXPERIMENT` code on the robot needs a second sandbox, and how you return structured errors vs raw tracebacks.

---

## Docs search: “distance-based” vs similarity — same thing

They embedded all **14 sections** of the OT-2 API docs with OpenAI **ada**. Planner issues `DOCUMENTATION <query>`. They embed the query with the same model. They pick the closest sections.

They write both “compute **similarity** with respect to a query” and “selected via a **distance-based** vector search” in the same paragraph. That is not two algorithms.

Embedding retrieval:

1. Each doc chunk → vector.
2. Query → vector.
3. Rank by how close the vectors are.

“Close” can be said as **high cosine similarity** or **low cosine/L2 distance**. For normalized vectors, L2 distance and cosine similarity are the same ranking. FAISS-style ANN is usually implemented as nearest-neighbor (distance). People still call the whole method “similarity search.”

They are **not** doing keyword (inverted-index / BM25) search for OT-2. Nature even contrasts those two IR families and says they took the vector-database one so synonyms work without query expansion.

**7800 token cap:** GPT-4 context was small. They pack nearest sections until the raw text hits 7800 tokens, one shot, so the Planner gets the docs in a single `DOCUMENTATION` turn. That is why they chunk by *section* and stop adding sections when the budget is full — not because 7800 is a magic IR hyperparameter.

This mattered for the **heater–shaker**: GPT-4 did not know the module (released after cutoff). Vector search pulled the right API page; the Planner then used the real method names.

---

## ECL / SLL — the paragraph that was confusing

Two robotic APIs in this paper. They are not the same problem.

**OT-2 (Opentrons).** Language = Python, which GPT-4 already writes. Docs still help (especially new hardware). Vector search over 14 API sections is enough.

**ECL (Emerald Cloud Lab).** Language = **SLL (Symbolic Lab Language)**. This is a custom DSL for a remote automated lab: things like `ExperimentHPLC[sample, Instrument -> Model[...], ...]`. GPT-4 was **not** trained on SLL. You cannot prompt “write an HPLC protocol” and expect valid SLL.

“A greater challenge… more diverse robotic platform” = ECL is not one Python package. It is a whole cloud lab with many instruments and a language the model has never seen. So they test whether **docs in context** can teach it that language.

Pipeline they actually ran:

1. **Prompt-to-function.** User: “analyse a mixture to see what is in it.” Docs searcher looks at ECL’s function guide, returns the right function names (e.g. `ExperimentHPLC`). Figure 3B / Appendix G: they claim it picks the right functions.
2. **Prompt-to-SLL.** Take the **raw docs for that one function** (options, instruments, parameters, `Models` / `Objects`). Stuff them through a **second GPT-4** whose only job is: keep the syntax, summarize the knobs, emit a **code block** using that function. That block goes back to the **Planner**. Planner fills in concrete options using whatever the Web searcher found (or ECL defaults). Nature: they actually executed generated HPLC SLL at ECL on a caffeine standard. Air bubble in the injection — they flag that cloud labs still need QC.
3. **Prompt-to-samples.** Separate index of **1,110** stock samples on ECL shelves. Query `"Acetonitrile"` → matching bottles. So the agent can name things that physically exist in that lab.

The second GPT-4 is a **docs compressor / syntax keeper**, not another scientist. SLL docs are long; the Planner cannot eat the whole ECL manual. Summarize one function → Planner writes the call.

Web searcher and docs searcher meet at the end: docs tell you *how to call* `ExperimentHPLC`; web search (or ECL software) tells you *which column / gradient / sample*.

---

## “Automatically generated outputs” — what that phrase means

Not a second model generating commentary.

When `PYTHON` or `EXPERIMENT` runs, the **runtime** produces text: traceback, stdout, empty output, robot error. That text is stuffed into the next user message. The Planner did not write it. A human did not type it. The tool did. That is “automatically generated.”

The interesting claim: GPT-4 can **use that text as a signal and rewrite the code**. Same pattern as a human staring at a stack trace.

They also did this on the robot: wrong heater–shaker method name → (in the integrated experiment they went to `DOCUMENTATION` rather than only the traceback) → fixed protocol.

---

## The SymPy → `print()` episode (Appendix D)

Task: calculate reagent masses for a small Suzuki mechanistic study.

1. Planner emits `PYTHON` with `from sympy import Eq, solve, symbols`. **SymPy is not in the Docker image.** Interpreter returns an `ImportError` traceback (automatically generated). Planner sees it: “I cannot use external libraries like Sympy.”
2. Rewrites in plain Python. Last line is the tuple `bromobenzene_mass, phenylboronic_acid_mass`. In a **REPL / Jupyter**, that would display. In a **script** (`python file.py` inside Docker), expressions that are not `print`ed produce **no stdout**. Tool returns empty.
3. Next user message in the appendix is explicit: `Python returned nothing. Use print() function.` So the harness (or they) **hinted**. The paper’s prose makes it sound like the model noticed on its own; the transcript shows a nudge on this third step.
4. Planner adds `print(...)`. Numbers come back. It continues.

Still a useful design: **execution feedback in the chat** + a tiny environment (no extra packages, non-interactive interpreter) forces the model to write boring, runnable scripts. Worth copying. Worth not overstating — the `print()` hint was in the user message.

---

## Decisions to steal later (not implementing now)

- **One planner, text commands, tool results as user messages.** Cheap. Matches the public 80-line loop.
- **One command per turn.** Prevents the model from firing `EXPERIMENT` before `GOOGLE`/`PYTHON` finish.
- **Fixed step budget (~10) + “do not ask the user.”** Stops premature stopping and infinite clarification.
- **LLM tools vs non-LLM tools.** Search/docs may use a model. Code and hardware execution must not.
- **Sandbox the interpreter.** Discuss Docker (or equivalent) before we ever `exec` model code.
- **Docs RAG for APIs the model does not know.** Embed chunks, retrieve by nearest neighbor, cap tokens so it fits one turn. Extra summarizer LLM if the API language is a DSL (SLL case).
- **Self-repair from tracebacks / empty stdout.** Maybe a thin wrapper that says “no stdout, use print” instead of hoping the model infers it.

---

## What they did *not* do

- No multi-agent debate / critic / tournament (that is later papers).
- No fine-tuning. Prompt + tools.
- Full prompts and the real lab stack are withheld. Reproduce from the toy loop + the paper, not from a complete codebase.
- OT-2 setup was not fully walk-away automated (human moved plates).
- Chemistry search is Google, not Reaxys/SciFinder. They say that would help multistep synthesis.
- Dual-use: they ran a safety appendix; that is why the real agent is not on GitHub.
