# Contexto Solver Progress And Decisions

This document records the implementation progress and design decisions made so
far. It is intended to help communicate the project state to the supervisors and to
preserve decisions that may matter for a future paper or report.

## Project Goal

The project studies automated strategies for solving Contexto-like word
guessing games. A game returns a semantic rank for each guessed word; lower
ranks indicate higher semantic closeness to the hidden target. The program
compares solvers that use:

- LLM-generated semantic hypotheses and candidate words.
- Embedding nearest-neighbor search.
- Local games where the ranking model is known.
- The real Contexto API where the ranking model is unknown.

## Implemented Components

### Stage 1: Local Contexto Game

Implemented:

- `EmbeddingModel` loads GloVe text embeddings.
- `LocalGame` ranks the full vocabulary by cosine similarity to a target word.
- `play.py` provides a manual terminal game loop.
- The local game follows the shared game interface:
  - `guess(word) -> int`
  - `total_guesses() -> int`
  - `best_so_far() -> tuple[str | None, int | None]`
  - `is_solved() -> bool`

Validation completed:

- Loaded `glove.6B.300d.txt`.
- Confirmed 400,000 vocabulary entries and 300-dimensional vectors.
- Confirmed nearest neighbors for `king` include `queen`, `prince`, and
  `monarch`.
- Confirmed `LocalGame(target="cat")` returns:
  - `cat -> 1`
  - `dog` closer than `car`
  - unknown words return `-1`

### Stage 2: LLM Evolutionary Solver

Implemented:

- `SolverLLM` runs an evolutionary loop over semantic hypotheses.
- `LLMClient` supports OpenAI and Anthropic.
- Prompts require raw JSON only.
- Guesses are filtered to one lowercase dictionary-style token.
- Invalid/unrecognized guesses are remembered and avoided in later prompts.
- Selection, mutation, crossover, local search, and elitism are represented in
  traces.
- Independent LLM calls are parallelized through `LLM_WORKERS` /
  `--llm-workers`.

Validation completed:

- LLM solver successfully solved a local game with target `cat`.
- LLM solver successfully solved real Contexto game 1314 after correcting the
  answer-rank normalization.

### Stage 3: Embedding Evolutionary Solver

Implemented:

- `SolverEmbedding` uses embedding nearest neighbors instead of an LLM.
- The embedding solver can run against local games and the real API.
- Reproducible seed controls were added:
  - `seed_count`
  - `active_count`
  - `neighbors_per_word`
  - `random_seed`

## Experiment Workflow

Implemented:

- `contexto_solver/experiment.py` runs batch local experiments.
- It supports:
  - aligned embedding experiments
  - non-aligned embedding experiments
  - embedding solver
  - LLM solver
  - multiple targets
  - repeated runs per target
  - JSON summary output
  - CSV summary output

Current scope:

- The batch runner focuses on local games.
- Real API batch experiments are intentionally deferred because they require
  careful handling of rate limits and archived game selection.

## Important Design Decisions

### 1. Shared Game Interface

Both local and real games expose the same interface. Solvers do not know whether
they are playing locally or against the API.

Research importance:

This isolates solver behavior from game implementation. It lets us test the
same algorithm under known-model and unknown-model conditions.

### 2. Rank Normalization

The local game uses rank `1` for the correct answer. The real Contexto API
returns distance/rank `0` for the answer. The API wrapper normalizes the real
API answer to rank `1`.

Research importance:

Without this normalization, solver stopping behavior differs across game
backends. Normalizing preserves a single game contract for fair comparisons.

### 3. Local Game Uses Full-Vocabulary Ranking

`LocalGame` precomputes cosine similarity between the target word and every
word in the embedding vocabulary, then assigns ranks.

Research importance:

This creates a deterministic offline approximation of Contexto where the hidden
ranking function is known. It enables unrestricted testing without API costs or
rate limits.

### 4. Aligned vs Non-Aligned Embeddings

The project now supports separate embedding paths for:

- The local game's ranking model.
- The embedding solver's search model.

Research importance:

Aligned experiments test an upper-bound condition where the solver has access to
the same semantic geometry as the game. Non-aligned experiments test robustness
when the solver's semantic geometry differs from the game's hidden geometry.
This mirrors the real Contexto API condition.

### 5. LLM Solver As A Semantic Hypothesis Generator

The LLM solver does not access embedding vectors directly. It proposes
categories and words based on language knowledge and feedback from ranks.

Research importance:

This allows a comparison between explicit vector-neighbor search and LLM-guided
semantic exploration. The comparison addresses whether the LLM contributes
semantic reasoning beyond nearest-neighbor similarity.

### 6. JSON Traces As Explainability Artifacts

Every run writes readable JSON traces with events such as `INIT`, `GUESS`,
`SELECT`, `MUTATE`, `CROSSOVER`, `LOCAL_SEARCH`, `SOLVED`, and `FAILED`.

Research importance:

The traces make the solver's search path auditable. They support qualitative
analysis of how hypotheses evolve, why candidates were chosen, and where search
converged or failed.

### 7. API Rate Limit Applies Only To Real Contexto

Local games have no per-guess delay. The rate limit exists only in `ContextoAPI`
for responsible real API use.

Research importance:

This keeps local experiments fast and avoids conflating algorithm performance
with external service rate limits.

### 8. Parallel LLM Calls

Independent LLM calls for candidate generation and mutation are parallelized.

Research importance:

This reduces wall-clock time for LLM experiments without changing the number of
game guesses. It improves experimental throughput while preserving the search
algorithm's observable behavior.

### 9. Local vs Online Performance Gap

Recent runs suggest the LLM evolutionary solver performs better against the
online Contexto API than against the local GloVe game, even though the LLM solver
does not directly access either backend's embeddings.

Research importance:

This may indicate that GPT-5.4-mini's implicit semantic geometry aligns better
with the real Contexto ranking model than with GloVe. If this pattern holds over
more targets, it becomes an important result about implicit manifold alignment
between LLMs and word-similarity models.

### 10. Local Search Needs A Fallback Budget

Local search is helpful when the current best word is extremely close, such as
`dog` at rank 2 leading to `cat`. It is less reliable for moderately close clues
such as ranks 30-100, where it can keep proposing words near the same semantic
area without reaching the target.

Research importance:

The solver needs a better exploration/exploitation balance. A likely next step
is to give local search a fixed guess budget and return to category exploration
when it stalls.

## Current Performance Snapshot

Online game + LLM evolutionary solver:

- Before active-cap/dedup/pivot fixes: roughly 500 guesses in difficult runs.
- After fixes: often under 200 guesses in the first post-fix verification set.
- Latest additional run: API game `1323` solved as `sponges` in 254 guesses over
  15 generations.

Local GloVe game + LLM evolutionary solver:

- Before fixes: 600+ guesses and 20+ generations in difficult runs.
- After fixes: roughly 300 guesses in better verification runs.
- Latest `notorious` runs still failed within 20 generations, ending at `gang`
  rank 4 with 380-387 guesses.

Current interpretation:

- Variance is high across targets.
- The online API appears easier for the LLM solver than the local GloVe game.
- The current evolutionary loop can still converge too narrowly once it finds a
  strong but incomplete clue.

## Current Commands

Manual local play:

```powershell
python play.py cat
```

Single local LLM run:

```powershell
python main.py --game local --target cat --solver llm --llm-workers 8
```

Single local aligned embedding run:

```powershell
python main.py --game local --target cat --solver embedding --random-seed 123
```

Single local non-aligned embedding run:

```powershell
python main.py --game local --target cat --solver embedding ^
  --game-embedding-path data/glove.6B.300d.txt ^
  --solver-embedding-path data/other_vectors.txt
```

Batch aligned embedding experiment:

```powershell
python -m contexto_solver.experiment --targets cat,dog,ivory --mode aligned --solver embedding --random-seed 123
```

Batch non-aligned embedding experiment:

```powershell
python -m contexto_solver.experiment --targets cat,dog,ivory --mode non_aligned --solver embedding ^
  --game-embedding-path data/glove.6B.300d.txt ^
  --solver-embedding-path data/other_vectors.txt
```

## Remaining Work

- Run more local-game targets to test whether performance stabilizes across
  target words.
- Add a local-search fallback budget so stalled local search returns to broader
  category exploration.
- Improve exploration/exploitation balance and diversity maintenance without
  relying on a stronger LLM model.
- Run aligned and non-aligned embedding experiments with real generation budgets.
- Compare LLM, aligned embedding, non-aligned embedding, and API-unknown runs on
  the same target set.
- Add automated tests that use a tiny fixture embedding file to avoid loading
  full GloVe in CI.
