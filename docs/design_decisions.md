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
semantic stagnation. Its effectiveness is currently being evaluated in a paired
pivot-on/off matrix; no final result should be claimed until that analysis is
complete.

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

## Future Diversity Direction

Observation: the project has discussed exploration/exploitation balance and
diversity maintenance, and the solver contains a fresh-diversity pivot path.

Status: there is no explicit MAP-Elites/archive design or implementation yet.
Any MAP-Elites-inspired archive should be treated as future work pending the
pivot-matrix results.
