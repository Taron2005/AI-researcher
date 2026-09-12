# DO Challenge & Adjacent Agentic-Systems Benchmarks — Research Notes

**Source paper:** Smbatyan, Ghukasyan, Aghajanyan et al., *"Can AI Agents Design and Implement Drug Discovery Pipelines?"* Deep Origin, 2025. arXiv: [2504.19912](https://arxiv.org/abs/2504.19912)
**Benchmark dataset (CC-BY-4.0):** Zenodo record [10.5281/zenodo.15296510](https://zenodo.org/records/15296510)
**Client + server code:** [github.com/deeporiginbio/do-events-challenge2025](https://github.com/deeporiginbio/do-events-challenge2025)

---

## 1. The core task, in the authors' words

> "In this work, we introduce DO Challenge, a novel benchmark specifically designed to evaluate the comprehensive capabilities of autonomous agentic systems in drug discovery. Unlike existing benchmarks focused on isolated tasks, DO Challenge presents a single, integrated challenge inspired by virtual screening, requiring agents to identify promising candidates from a chemical library of one million molecular structures. To succeed, agents must autonomously develop and execute strategies that involve **exploring chemical space, selecting predictive models, balancing multiple objectives, and managing limited resources** — mirroring the complex, resource-constrained decision-making environment of pharmaceutical research."

This is the paper's central thesis: existing benchmarks (§6–8 below) each isolate one skill. DO Challenge forces all four to happen together, in one task, with nothing telling the agent which skill matters when.

### 1.1 Breaking down the four demands

- **Exploring chemical space.** The agent starts with 1M unlabeled 3D molecular conformations and no target information beyond "find the ones with the highest score." It must decide *where* to spend a limited labeling budget — random sampling, clustering, similarity search, or active learning were the tools the paper's top solutions actually used.
- **Selecting predictive models.** Once some labels exist, the agent must pick a model class suited to a small-to-medium labeled set predicting a physically grounded, 3D-structure-dependent score — anywhere from gradient-boosted trees on fingerprints to GNNs on 3D graphs.
- **Balancing multiple objectives.** The label (DO Score) is a difference: strong binding to the therapeutic target *minus* the worst binding across three ADMET-liability proteins. A model that ignores the ADMET penalty will systematically pick bad structures.
- **Managing limited resources.** Only 100,000 of 1,000,000 structures can ever be labeled, and only 3 submissions of 3,000 structures are scored. Every label request and every submission is an irreversible spend.

## 2. Task mechanics (resource-constraint numbers)

| Constraint | Value |
|---|---|
| Dataset size | 1,000,000 molecular conformations (SDF files), unlabeled at start |
| Label budget | ≤100,000 structures (10% of the dataset) — requestable all at once or in batches |
| Submissions | Exactly 3 attempts, 3,000 structure IDs each; best of the 3 counts |
| Scoring | `Score = |Submission ∩ True Top 1000| / 1000 × 100%` |
| Evaluation setups | (i) 10-hour time-limited, (ii) unrestricted, (iii) 1-week post-challenge extension |

> "During the development of the solution, the agents are allowed to request only 10% of the true values of DO Score for structures of their choice, and only 3 submissions can be presented for evaluation, simulating a resource-constrained environment. Performance is measured as the percentage overlap between the set of actual top 1000 molecular structures in the challenge dataset and the set of structures selected by the agents. In addition, benchmark performance can be reported in time-constrained (10 hours for development and submission) and unrestricted setups."

**Note on the 10-hour number:** that's the time limit for one specific *evaluation condition* inside the paper's own competition (DO Challenge 2025) — it bounds how long a human team or an agent gets to build *and* submit a solution to *that* benchmark. It is not a property of the QM8 assignment, and it doesn't mean an agentic system inherently needs 10 hours of wall-clock runtime or expensive infra to be valid. The only real constraints on the QM8 work are the Sep 14 deadline and whatever compute budget is self-imposed.

## 3. Can this actually be run independently?

**Yes.** Unlike most benchmark papers, DO Challenge shipped everything needed to reproduce it standalone:

- **Zenodo record** (CC-BY-4.0, reuse permitted with attribution): `ligand_structures_data.tar.gz` (964 MB, all 1M SDF conformations) + `labels.pkl` (ground-truth DO Scores, ID → score) + `task.md` / `task-2025.md` (exact task text given to agents/humans).
- **`deeporiginbio/do-events-challenge2025`** on GitHub ships both a `/client` *and* a `/server` — the actual label-request/submission-scoring service used in the competition, not just a client stub. That means the whole loop can be self-hosted: serve `labels.pkl` behind the same API surface, point an agent harness's client at it, and get real overlap-with-top-1000 scores without needing Deep Origin's original infrastructure.

**Is it needed for the QM8 deliverable?** No. The assignment is QM8 — a fully-labeled, non-resource-constrained dataset. DO Challenge is Deep Origin's own harder, closed-flavor benchmark from this paper, not what was asked for. It's optional extra context: running the same harness against a subsample of DO Challenge later, after the QM8 deliverable is solid, would be a signal that the agent design generalizes — but it's not part of what was asked, and skipping it costs nothing.

## 4. The four success factors the authors identified

> 1. **Strategic structure selection**: Employ sophisticated structure selection strategies (such as active learning, clustering, or similarity-based filtering).
> 2. **Spatial-relational neural networks**: The adoption of neural network architectures such as Graph Neural Networks (GNNs), attention-based architectures, 3D CNNs, or their variants, specifically designed to capture spatial relationships and structural information within molecular conformations.
> 3. **Position non-invariance**: Utilize features that are not invariant to translation and rotation of the structure. It is important to note that the task description openly specified DO score's sensitivity towards atom position changes, but not all solutions accounted for this aspect of the label.
> 4. **Strategic submitting**: Combine true labels and model predictions intelligently; leverage the provided submission count and use the outcomes of previous submissions to enhance subsequent submissions.

This is close to the single most load-bearing paragraph in the paper. Two reasons why.

**First, it's measured, not asserted.** Every human and agent solution was coded against these four binary criteria and correlated with benchmark score:

| Factor | Corr. (10h limit) | Corr. (unrestricted) |
|---|---|---|
| Strategic structure selection | 0.53 | 0.28 |
| Spatial-relational neural networks | 0.25 | 0.36 |
| Position non-invariance | 0.56 | 0.33 |
| Strategic submitting | −0.07 | 0.35 |

Structure selection and position-sensitivity dominate under time pressure; all four matter roughly equally once time pressure is removed. Strategic submitting is *negatively* correlated under the 10-hour limit — the authors' reading is that agents capable enough to plan a multi-submission strategy were also spending scarce time on that planning instead of on a better single submission, so it's a second-order optimization that only pays off once the first three are already handled.

**Second, factor 3 is a documented, named failure mode, not just a missing nice-to-have.** The task text *explicitly told* solvers that DO Score is sensitive to translation/rotation of the 3D structure — and the paper's failure-mode analysis found agents still frequently chose invariant or equivariant featurizations anyway, sometimes reasoning "this should be invariant" in direct contradiction of the stated spec. That's a spec-reading failure, not a modeling limitation.

**How this maps to the QM8 work — and where it doesn't:**
- Factors 1 and 2 (strategic selection, spatial-relational architectures) transfer directly: QM8's labels (TDDFT/CC2 excitation energies and oscillator strengths) are also 3D-structure-dependent, and GNN/MPNN-style architectures are the literature-standard for exactly that reason.
- Factor 3 is not a live trap in QM8 the way it is here — QM8 doesn't hide a translation/rotation sensitivity behind an explicit warning, it just supplies a fixed 3D geometry per molecule and rewards position-aware modeling with better accuracy. Still worth checking that any QM8 agent doesn't default to discarding 3D coordinates.
- Factor 4 (strategic submitting) doesn't apply to QM8 at all — there's no submission-budget game, since the whole dataset is labeled from the start.

## 5. Benchmarks sharing DO Challenge's "agent doing ML research" framing, generically (not drug-discovery-specific)

| Benchmark | What it tests | Worth evaluating on? |
|---|---|---|
| **MLGym** (arXiv:[2502.14499](https://arxiv.org/abs/2502.14499), Meta FAIR) | A Gym-style RL environment plus **MLGym-Bench**: 13 open-ended AI-research tasks across CV, NLP, RL, and game theory (generate hypotheses, write code, train models, analyze, iterate). Frontier models mostly improve on given baselines via better hyperparameters, rarely via genuinely novel methods. | Not needed. Integrating with MLGym's Gym API is a non-trivial harness adaptation for a benchmark with no chemistry flavor at all — low payoff against a Sep 14 deadline. |
| **MLE-Bench** (arXiv:[2410.07095](https://arxiv.org/abs/2410.07095), OpenAI) | 75 real Kaggle competitions (CV, NLP, tabular, signal). Agents get raw competition data and must produce a submission, scored against real Kaggle leaderboards/medal thresholds. Best reported agent (o1-preview + AIDE scaffold) hit medal level in ~17% of competitions. | Closest in *kind* to this work — same explore → model → submit shape. If a bonus generalization data point is wanted later, this is the cheapest one to bolt a harness onto. Not required. |
| **SWE-bench** (arXiv:[2310.06770](https://arxiv.org/abs/2310.06770)) | ~2,000+ real GitHub issues; agent must produce a patch that makes the repo's held-out test suite pass. Pure software-engineering correctness, no modeling involved. | No — different capability, no natural connection point. |
| **MLAgentBench** (Huang et al. 2024, ICML) | Agent gets a workspace with a dataset and starter code and must improve a specified metric within a runtime/compute budget, tracked action-by-action. Closer to "ML engineer iterating on one dataset" than MLE-Bench's competition framing. | Same story as MLE-Bench — plausible cheap add-on later, not required now. |
| **RE-Bench** (arXiv:[2411.15114](https://arxiv.org/abs/2411.15114), METR) | 7 hand-built, open-ended **ML research-engineering** environments (e.g. fit a scaling law, write a faster GPU kernel), compared against 71 real 8-hour human-expert attempts. Best AI agents beat humans at short time budgets (2h) but humans pull ahead by 32h. | No — systems/infra-level AI R&D, a different skill set from molecular property prediction. |

## 6. General-purpose / professional-task agent benchmarks (landscape completeness — not directly relevant here)

| Benchmark | What it tests |
|---|---|
| **TheAgentCompany** (arXiv:[2412.14161](https://arxiv.org/abs/2412.14161), CMU) | 175 long-horizon "digital worker" tasks inside a self-hosted simulated software company (real services: GitLab, RocketChat, ownCloud, a project tracker) plus LLM-driven simulated coworkers. Best model (Gemini 2.5 Pro) completed only ~30% of tasks autonomously. |
| **PlanBench** (Valmeekam et al. 2023, NeurIPS) | Tests **plan generation and reasoning about change** in isolation (classic blocksworld-style domains) — whether an LLM can produce a *correct plan*, not whether it can carry that plan out in a real environment. |
| **BrowseComp** (arXiv:[2504.12516](https://arxiv.org/abs/2504.12516), OpenAI) | 1,266 deliberately hard "needle-in-haystack" web-research questions, constructed so answers are verifiable but not surfaced by ordinary search — tests persistence and multi-hop reasoning in browsing agents specifically. |

None of these three need evaluation here — they test communication-with-simulated-humans, symbolic planning, and web search respectively, none of which is "agents doing ML research on a dataset." The DO Challenge paper cites them only to establish that agentic-capability benchmarks span many flavors, and drug discovery specifically was under-covered by all of them.

## 7. Scientific-research-automation benchmarks (closer to this task, still not drug-discovery-specific)

| Benchmark | What it tests | Where it differs from DO Challenge |
|---|---|---|
| **ScienceAgentBench** (arXiv:[2410.05080](https://arxiv.org/abs/2410.05080), OSU) | 102 tasks drawn from 44 real peer-reviewed papers across 4 disciplines (including a computational-chemistry task on the ClinTox dataset). Agent must produce one self-contained Python program reproducing a specific analysis step. Best agents solve only ~32–34%, even with expert-provided hints. | Tests one *isolated analysis step* per task against a known-correct reference output — deliberately not an open-ended, multi-stage discovery process under uncertainty. This is the specific gap the DO Challenge authors name: it "does not specifically tackle the nuanced demands and uncertainties inherent in drug discovery." |
| **CORE-Bench** (arXiv:[2409.11363](https://arxiv.org/abs/2409.11363), Princeton) | 270 tasks from 90 published papers (CS, social science, medicine). Agent is handed the paper **and its original code repository**, and must install dependencies, run the code, and correctly answer questions about the reproduced outputs. | Measures whether an agent can reproduce a *known, already-solved* result given the solution's own code — useful for reproducibility auditing, but the opposite problem from DO Challenge, where there's no existing code and no known-correct pipeline. |
| **PaperBench** (arXiv:[2504.01848](https://arxiv.org/abs/2504.01848), OpenAI) | Agent gets 20 ICML 2024 Spotlight/Oral papers and must replicate each **from scratch** — no starter code — against an 8,316-item hierarchical grading rubric co-developed with the original authors. Best agent (Claude 3.5 Sonnet + open scaffold) scored 21.0% average vs. 41.4% for recruited ML PhDs. | Closer to DO Challenge in that no code is handed over, but the target is still a *known* result (the original paper's reported numbers) the agent races to match. No exploration under genuine uncertainty about what the right answer even is — DO Challenge's central twist is that nobody, including the benchmark's own designers, can check against the true top-1000 during development. |

## 8. Drug-discovery / molecular-ML benchmarks — the "isolated tasks" DO Challenge argues against

The set the paper positions itself against most directly, since these are the actual domain-specific predecessors:

| Benchmark | What it is | Authors' stated gap |
|---|---|---|
| **Therapeutics Data Commons (TDC)** (Huang et al. 2021) | 66+ datasets across 22+ tasks spanning the entire drug-discovery pipeline (target ID, ADMET, generation, etc.) — the broadest single resource of the group. | Grouped with DrugOOD and CARA: "often treat these tasks independently rather than assessing the comprehensive, integrated capabilities required by fully autonomous agents." Breadth of *tasks*, not integration of *decision-making across* them. |
| **DrugOOD** (arXiv:[2201.09637](https://arxiv.org/abs/2201.09637)) | Automated dataset curator + benchmark specifically for **out-of-distribution generalization** in binding-affinity prediction, built from ChEMBL with annotated domain/noise levels. | Same critique — tests generalization on one predictive task, not strategic decision-making across an end-to-end pipeline. |
| **CARA** (Tian et al. 2024, *Communications Chemistry*) | Real (not synthetic) wet-lab compound-activity data from ChEMBL, organized by assay, explicitly split into virtual-screening vs. lead-optimization task types to avoid the overoptimistic scores random splits give. | Same critique — realistic data and splits, still a single predictive task, not an integrated agentic pipeline. |
| **GuacaMol** (Brown et al. 2019) | Benchmark suite for **de novo molecular generation** — goal-directed design (optimize a property) and distribution-learning (generate realistic molecules) tasks. | Grouped with MoleculeNet/MolGym: "address individual components such as molecule design or property prediction" — generation is a different, narrower skill than the *screening/selection* task DO Challenge poses. |
| **MoleculeNet** (Wu et al. 2018) | The standard broad benchmark suite for **molecular property prediction** — this is where QM8 itself comes from. Defines the splits/metrics used across the field. | Same critique — a fixed, fully-labeled property-prediction task per dataset, no resource constraints, no agentic decision layer on top. |
| **MolGym** (Simm, Pinsler & Hernández-Lobato, 2020) | RL environment for **building molecules atom-by-atom in 3D**, with reward from real quantum-chemical energy calculations, in a translation/rotation-invariant state-action space. | Same critique — tests an agent's ability to *construct* a molecule under a QM-derived reward, not to *select* promising candidates from an existing library under a limited labeling budget. |

**Worth noting explicitly:** by the DO Challenge paper's own taxonomy, the QM8 assignment sits squarely in this last category — MoleculeNet-style, fully-labeled, single-task, no resource constraints, no agentic decision layer forced by the dataset itself. Whatever "agentic research capability" Deep Origin is actually scoring has to come from *how the multi-agent process is designed* around QM8 (hypothesis generation, experiment design, iteration, honest reporting of failures) — not from the dataset forcing that behavior the way DO Challenge's label-budget-and-submission-limit mechanic does. That's the design lesson worth carrying over: consider imposing some DO-Challenge-style constraints deliberately (a self-set compute/query budget, a cap on how many "experiments" the agent team can run) purely to force the same strategic, resource-aware behavior the paper is testing for, since QM8 alone won't impose it.
