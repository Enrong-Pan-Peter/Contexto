# Design Decisions

This document records algorithmic and experimental design decisions that are
important for understanding or extending the Contexto solver. Low-level module
structure belongs in `docs/architecture.md`; run-level evidence belongs in
`docs/experiment_log.md`; paper-facing empirical claims belong in
`docs/findings.md`.

## Shared Game Interface

Decision: both local and real games expose the same small interface:

```python
guess(word) -> int
total_guesses() -> int
best_so_far() -> tuple[str | None, int | None]
is_solved() -> bool
```

Rationale: solvers should not need to know whether ranks come from the local
GloVe game or the real Contexto API. This keeps solver behavior comparable
across known-model and unknown-model settings.

Research relevance: the same solver can be evaluated against controlled local
games and external API games without changing the search algorithm.

## Rank Normalization

Decision: all game backends expose the correct answer as rank `1`.

Rationale: the local game naturally ranks the target as `1`, while the real
Contexto API returns answer distance/rank `0`. The API wrapper normalizes this
so solver stopping behavior is backend-independent.

Research relevance: rank normalization prevents backend-specific solved logic
from confounding comparisons.

## Local Embedding Game

Decision: the local game precomputes cosine similarity from the target word to
the configured embedding vocabulary and assigns ranks over that vocabulary.

Rationale: this creates a deterministic offline approximation of Contexto with a
known ranking function.

Research relevance: local games allow repeated experiments without API costs or
rate limits, while still preserving a black-box rank-feedback interface for the
solver. Replacing GloVe with a MiniLM cache as the default backend gives the
local game a stronger modern semantic space without changing the game contract.

## LLM Solver as Backend-Agnostic Hypothesis Generator

Decision: the LLM solver proposes semantic categories and candidate words using
only rank feedback and trace state. It does not inspect local embedding vectors
or target internals.

Rationale: the goal is to test whether an LLM can guide semantic search without
knowing the game's embedding model.

Research relevance: this enables comparison between language-model-guided search
and explicit embedding-neighbor search.

## Embedding Baseline and Alignment Modes

Decision: the project supports separate embedding paths for the local game and
the embedding solver.

Rationale: when both paths match, the embedding solver is aligned with the game
geometry. When they differ, the solver searches with a different geometry from
the game backend.

Research relevance: aligned experiments approximate an upper-bound specialist
condition; non-aligned experiments better resemble unknown-backend settings.
The current embedding set is GloVe as a legacy/static baseline,
`all-MiniLM-L6-v2` as the lightweight default local backend after cache
generation, and `all-mpnet-base-v2` as a heavier quality-oriented option. The
transformer models are precomputed into static caches so aligned and
non-aligned comparisons use the same runtime interface.

## Traceability as an Explainability Mechanism

Decision: solver runs write readable JSON traces with events such as `INIT`,
`GUESS`, `SELECT`, `MUTATE`, `CROSSOVER`, `LOCAL_SEARCH`, `PIVOT_TRIGGERED`,
`SOLVED`, and `FAILED`.

Rationale: the search process itself is important evidence. Trace logs make it
possible to inspect why the solver converged, stalled, or changed direction.

Research relevance: traces support qualitative analysis of algorithm behavior,
not just final solve rates.

## Active Hypothesis Cap and Deduplication

Decision: limit the number of active hypotheses that propose candidates each
generation and merge near-duplicate hypotheses.

Rationale: early LLM runs produced many redundant categories, spreading the
guess budget across similar directions.

Research relevance: this is an algorithmic control for search bloat. Existing
evidence is trace-based and small-sample, so it should be framed as a
stabilizing implementation decision rather than a fully quantified result.

## Divergent Mutation

Decision: mutation prompts ask for alternative interpretations of strong clues,
not only narrower subcategories.

Rationale: early runs showed the solver could overcommit to one sense of a
word, such as following `bite` into food-related meanings while missing animal
relations.

Research relevance: divergent mutation is an attempt to improve semantic
branching in a rank-feedback search space.

## LLM-Guided Local Search

Decision: when the solver reaches a strong clue, it asks the LLM for nearby
words across relation types such as synonyms, descriptors, collocations,
associated groups, and causes/effects.

Rationale: local search has solved cases where the best clue was extremely close
to the target, but can stall when the clue is only moderately close.

Research relevance: local search is best understood as exploitation around a
strong clue, not as access to the game's embedding model. It should not replace
broader exploration indefinitely.

## Invalid-Guess and Single-Token Filtering

Decision: prompts require one lowercase dictionary-style word, and solver-side
filtering rejects phrases, hyphenated words, punctuation, and repeated invalid
guesses.

Rationale: Contexto rejects multi-word or malformed guesses, and early runs
encountered invalid outputs.

Research relevance: this is mostly engineering hygiene, but it protects the
validity of guess-count measurements by reducing avoidable invalid submissions.

## Singular/Plural Family Filtering

Decision: obvious singular/plural variants are treated as redundant during
candidate acceptance.

Rationale: `herbaceous` traces showed `shrub -> shrubs` consuming search effort
without changing the semantic direction.

Research relevance: this prevents morphological duplicates from being counted as
meaningful exploration. It is motivated by qualitative trace evidence.

## Stall Pivot Mechanism

Decision: a stall detector can trigger LLM-backed pivot operators:

- Morphological or lexical expansion.
- Register shift.
- Adjacent-category jump.
- Fresh diversity after a full pivot cycle.

Rationale: several difficult targets produced near-target stagnation where the
solver found a strong clue but failed to identify the target relation.

Research relevance: the pivot mechanism is a concrete intervention for local
semantic stagnation. The local and HPC pivot matrices now provide batch-level
evidence that pivots improve aggregate speed and reduce failed-run variance, but
they do not reliably solve `notorious`; target-level claims remain limited by
n=5 per target. See `docs/findings.md` and `docs/experiment_log.md` for evidence
and interpretation.

## Self-Adaptive Mutation Operators

Decision: `ea_llm_self_adaptive` gives each hypothesis a four-component sigma
vector over mutation operators: `s_mutation`, `m_mutation`, `ml_mutation`, and
`l_mutation`.

Rationale: the four operators are categorical semantic moves rather than a
single continuous step size. `s_mutation` refines the current neighborhood,
`m_mutation` reframes a clue, `ml_mutation` moves to an adjacent category, and
`l_mutation` opens a broader fresh direction. These operator semantics are
implemented in the four prompt constants in
[`contexto_solver/llm_client.py`](../contexto_solver/llm_client.py) and mapped
to operator IDs in
[`contexto_solver/operators.py`](../contexto_solver/operators.py).

Perturbation rule: children inherit a Dirichlet-perturbed copy of the parent's
sigma, with a floor to keep every operator available. The original supervisor
discussion framed the adaptation idea in relation to log-normal self-adaptive
evolution strategies, but the implemented sigma is a probability vector on the
four-simplex. A multiplicative log-normal rule would require additional
renormalization to return to the simplex; `Dirichlet(alpha * parent_sigma)`
keeps the draw on the simplex directly, with `alpha` controlling drift
magnitude. The implemented rule is in
[`contexto_solver/operators.py`](../contexto_solver/operators.py).

Prompting constraint: sigma is backend metadata only. The LLM sees the selected
operator's prompt, but never sees sigma values, probabilities, or the list of
operator alternatives. This keeps stochasticity in the backend sampler rather
than asking the LLM to comply with hidden probabilities. Prompt-leakage checks
guard this contract in
[`contexto_solver/operators.py`](../contexto_solver/operators.py) and are
exercised by
[`tests/test_self_adaptive_operators.py`](../tests/test_self_adaptive_operators.py).

Crossover decision: in the self-adaptive method, crossover sigma is now a
Dirichlet-perturbed average of the two parent sigmas rather than a uniform reset.
This was motivated by **Single-run observation** from
[`ea_llm_self_adaptive_local_notorious_20260522_035806.json`](../traces/ea_llm_self_adaptive_local_notorious_20260522_035806.json):
the best hypothesis reached `gangster` rank 4, but the inspection lineage
terminated at a crossover child with uniform sigma, making the adaptive lineage
opaque. The method-local fix is implemented in
[`contexto_solver/methods/ea_llm_self_adaptive.py`](../contexto_solver/methods/ea_llm_self_adaptive.py).

Local-search decision: local search is disabled by default in adaptive mode.
The mechanism is code-confirmed in
[`contexto_solver/methods/ea_core.py`](../contexto_solver/methods/ea_core.py):
base `_local_search()` constructs a fresh `Hypothesis` without passing sigma, so
it receives the default uniform sigma and can later compete as a mutation
parent. **Single-run observation** from
[`ea_llm_self_adaptive_local_superficial_20260525_194148.json`](../traces/ea_llm_self_adaptive_local_superficial_20260525_194148.json)
showed local-search hypotheses entering adaptive selection and producing orphan
parent IDs in the inspection script because local-search hypothesis records were
not serialized. The adaptive override and `LOCAL_SEARCH_DISABLED` trace event
are implemented in
[`contexto_solver/methods/ea_llm_self_adaptive.py`](../contexto_solver/methods/ea_llm_self_adaptive.py).

Population decision: self-adaptive runs now default to `mu=15` and 15 initial
categories, while the `mu=5` baseline remains reachable with
`SELF_ADAPTIVE_MU=5` and `SELF_ADAPTIVE_INITIAL_CATEGORIES=6` or another
explicit comparison setting. This follows Ting's recommendation to give the
adaptive mechanism a larger population and matches the sigma-SA-ES intuition
that selection needs enough competing lineages to discriminate productive sigma
profiles. It is also motivated by the observed high between-seed sigma variance
at `mu=5`, where a small active set can over-amplify early stochastic lineage
differences.

Research relevance: this creates an evolutionary mechanism for adapting the
exploration scale without adding another LLM-visible decision process. Current
evidence is trace-level and small-sample; any performance claim requires
repeated runs or batch analysis. See
[`docs/experiment_log.md`](experiment_log.md) for run evidence and
[`docs/findings.md`](findings.md) for evidence-quality labels.

Open design direction after pooled MAP-Elites coupling analysis: adaptive
operator selection (AOS) should be evaluated as an alternative credit-assignment
mechanism. In the current self-adaptive machinery, a child inherits
`Dirichlet(alpha * parent_sigma)` regardless of which operator produced the
child's word. A win from `s_mutation` therefore carries the parent's sigma into
the archive rather than crediting `s_mutation` itself. The 2026-06-08 pooled
MAP-Elites analysis found a clean per-operator fitness gradient favoring
`s_mutation` over `l_mutation`, while the inherited sigma mechanism cannot
directly assign reward to the fired operator. A reward-windowed AOS variant would
credit the operator that fired by its observed reward, so it can capture "small
mutation wins" directly. This is an open direction, not a committed design
change.

Control priority: the frozen-sigma / random-sigma comparison is now more than a
neutral sanity check. It should test whether sigma adaptation causally
misallocates effort relative to the observed operator-fitness gradient. The
informed comparison profile is a static or AOS policy that gives more sampling
mass to small mutation, while still retaining some larger jumps for exploration.

## Ollama Local LLM Backend

Decision: `LLMClient` supports Ollama through its OpenAI-compatible local
endpoint, with `--provider ollama` and `--ollama-model`.

Rationale: local LLM execution avoids cloud quota limits and supports long
overnight experiments.

Research relevance: this makes repeated local evaluation more practical, but
model-specific results should be reported separately from cloud LLM results.

## Batch Experiment Checkpointing and Analysis

Decision: the batch experiment runner writes JSON/CSV summaries after each
completed run and supports `--resume`. A separate analysis script performs
paired pivot-matrix analysis.

Rationale: long local LLM runs can take hours and may be interrupted. Analysis
should be separated from the solver and batch runner.

Research relevance: the current evaluation workflow supports repeated target
runs, provider/model metadata, per-target breakdowns, solved-only guess
statistics, unsolved best-rank summaries, Wilcoxon signed-rank tests, and
Cliff's delta.

## Trace Visualization as Post-Hoc Analysis

Decision: trajectory plots live in `contexto_solver.plot_trajectory` as a
standalone analysis script, separate from solvers, methods, game backends, and
the trace schema.

Rationale: visualizations are useful for diagnosing search behavior, but they
should not affect solver decisions or require new fields in existing traces.
The plotting module therefore reconstructs best-so-far, active-hypothesis, and
population-level trajectories from already-recorded events.

Projection rationale: PCA is deterministic and reports explained variance, but
single-run checks on `superficial` showed low two-component variance in the
target neighborhood. UMAP and PaCMAP were added as qualitative projection
options for inspection, with fixed `random_state=42` and a fit-then-transform
workflow. The plotting code projects each unique word once and reuses that
coordinate so repeated words land at the same position.

Rank and distance rationale: rank trajectories preserve the game-facing signal
without requiring embeddings. Cosine-distance trajectories use the embedding
model that produced the local trace to show whether rank improvements correspond
to movement toward the target in that model's geometry.

Research relevance: these plots are diagnostic evidence for individual runs and
small comparisons, not performance estimates by themselves. Claims based on
figures should cite the underlying traces and distinguish qualitative single-run
inspection from repeated-run or batch-level summaries.

## Future Diversity Direction

Observation: the project has discussed exploration/exploitation balance and
diversity maintenance, and the solver contains a fresh-diversity pivot path.

Status: an explicit archive-based method now exists (`ea_llm_map_elites`, see
below). The original "future work" note is kept here for history; the MAP-Elites
selection layer is no longer purely speculative.

## MAP-Elites Archive Selection (`ea_llm_map_elites`)

Decision: add `ea_llm_map_elites`, a MAP-Elites variant of
`ea_llm_self_adaptive` that replaces top-mu selection with an archive over a
`5x5` grid of behavior cells. Two behavior axes are used: concreteness
(concrete/physical to abstract/conceptual) and specificity (general to
specific). Each cell holds at most one elite; per-cell competition keeps the
better-ranked hypothesis. The sigma self-adaptation mechanism is inherited
unchanged from the parent method.

Rationale (selection-layer diversity fix): earlier batches indicated the
diversity problem is at the selection layer, where top-mu/half selection
collapses lineages onto the current best region. An archive enforces structural
diversity: a fresh-jump child with a mediocre global rank still survives if it
lands in an empty cell or beats that cell's incumbent. This decouples "is this
the global best" from "is this the best example of this kind of hypothesis."

LLM-driven placement over embedding centroids: placement uses a single LLM call
with anchored scales rather than embedding-centroid math. This keeps the method
backend-agnostic and compatible with the real Contexto API, where solver-side
embeddings of the target neighborhood are not available. Anchored scales (for
example `0.00: rock ... 1.00: freedom` for concreteness) give the LLM a stable
frame of reference and make placements reproducible; anchors live in
`MAPELITES_ANCHORS_*` config and are hashed into the placement cache key so any
anchor change invalidates stale cache entries.

First axis pair choice: concreteness and specificity were chosen as the first
behavior descriptors because they are largely orthogonal, intuitive for an LLM
to rate on a `0-1` scale, and span the kinds of semantic moves the operators
already make (narrowing/refining vs. reframing/abstracting).

Override strategy: the method overrides `initialize()` and `run_generation()`
whole-cloth instead of hooking individual base steps. Because the base EA loop
is bypassed, per-hypothesis multi-candidate generation, top-mu/half selection,
the post-generation hook, and deduplication are all inactive without separate
overrides, matching canonical MAP-Elites where placement and fitness are
immutable post-creation. Each hypothesis is created with exactly one
`best_word`.

Research relevance: this is a structural diversity mechanism that is comparable
across local and real-API games. Any performance claim requires repeated runs;
the new trace events (`AXIS_DEFINITION`, `PLACEMENT`, `ARCHIVE_*`) are designed
to support later quality-diversity analysis (sigma heatmaps, archive scatter
plots), which is out of scope for the initial implementation.

## MAP-Elites Open Design Considerations After First Test Run

Evidence source:
[`traces/ea_llm_map_elites_local_superficial_20260606_015531.json`](../traces/ea_llm_map_elites_local_superficial_20260606_015531.json).
Evidence quality: single-run diagnostic observation only (`superficial`, seed 4,
Ollama `qwen3:14b`). These are open design considerations, not settled design
decisions.

Archive sampling tension: the first MAP-Elites test run surfaced a
coverage-versus-exploitation trade-off. Uniform archive sampling bought coverage
(19/25 occupied cells) but did not concentrate enough effort on the best cell to
convert `thin` rank 5 to rank 1 after generation 37. Two possible design forks
are now explicit but undecided: (a) quality-biased archive sampling, or (b)
stagnation-gated focused refinement on the best cell, reintroducing a
pivot-style local search only after best-rank stalling. Both require more runs
before changing the method.

One-word child pipeline: dropping multi-candidate generation and requesting one
word per child creates a clean one-to-one pipeline for successful children:
child -> valid `GUESS` -> `PLACEMENT` -> one archive outcome. In the first test
trace this produced 841 successful children, 841 valid guesses, 841 placements,
and 841 archive events. The same design makes generative exhaustion visible as a
collapse in children/gen: already-seen proposals are dropped in `_guess_first_valid`
before archive competition, so they reduce realized children rather than adding
duplicate archive contests.

Anchor recalibration: the specificity axis appears miscalibrated for the first
observed run. Six cells were empty and four of those were in the top specificity
row; no placement ever landed in any of the empty cells. Keep two explanations
separate: some sub-grid coverage may be structural for one target because useful
words occupy a limited behavioral region, while the top specificity row may also
be tunably too extreme because `northern cardinal` was not reached by
solver-generated words. A future recalibration should use the empirical
distribution of solver-generated words without assuming that all empty cells are
failures.

## MAP-Elites Sigma-Mode Control, Ranked Context, and Anchor Update

Anchor update: the default `MAPELITES_ANCHORS_CONCRETENESS` scale moved to
`rock, rain, music, fear, freedom` and `MAPELITES_ANCHORS_SPECIFICITY` to
`thing, animal, bird, songbird, sparrow`. The concreteness poles now use words
that read more cleanly as "physical object" through "pure abstraction", and the
specificity ceiling dropped from `northern cardinal` (a two-word proper-ish name
never reached by solver-generated words in the first run) to `sparrow`, a single
common word. Because anchors are hashed into the placement cache key, this change
automatically invalidates stale cache entries rather than mixing scales.

Sigma-mode flag rationale: the first runs could not separate "self-adaptive sigma
helps" from "the EA structure helps", because operator probabilities always
adapted. `MAPELITES_SIGMA_MODE` makes the sigma mechanism a controllable factor
without forking the method. The flag is read from config/env so a batch can sweep
arms by setting one environment variable per subprocess, leaving the rest of the
configuration identical.

Control design (four arms): `adaptive` is the current behavior; `frozen_uniform`
pins every child to the uniform operator prior; `frozen_fixed` pins every child
to a fixed non-uniform profile; `random` redraws operator probabilities from
`Dirichlet(1)` for every child. Together they bracket the adaptive mechanism
between "no adaptation, uniform", "no adaptation, deliberately skewed", and
"maximally noisy", so a difference in outcomes is attributable to how sigma is
assigned rather than to any other change. The mode is applied at all three
hypothesis-creation sites so an arm is internally consistent from generation 0;
`sample_operator(parent.sigma)` is intentionally left untouched, so the operator
that fires still follows the child's inherited sigma under every mode.

`frozen_fixed` profile choice: the default `MAPELITES_FROZEN_SIGMA` is
`[0.4, 0.3, 0.2, 0.1]` over `[s, m, ml, l]`, a monotone preference for smaller
mutations. This is a deliberate, easily-described skew (favor local refinement,
still allow occasional large jumps) that contrasts cleanly with the uniform arm,
not a tuned optimum. It is validated to a simplex on use, so a malformed override
fails fast rather than silently renormalizing.

Ranked-context design: `MAPELITES_RANKED_CONTEXT_K` (default `0`, off) optionally
injects the global top-K best-ranked guessed words into mutation prompts as a
shared `{ranked_context}` slot. It is shared so self-adaptive prompts remain
byte-identical (empty default), and populated only by MAP-Elites. The injected
content is game-rank feedback, not sigma, so it does not weaken the sigma-leakage
invariant; consistent with the existing `{all_guesses}` slot, the words are not
reserved-substring filtered.
