# Project Architecture

This document records the current architecture of the Contexto solver project.
Use it as the primary reference before creating, debugging, or refactoring code
so updates remain consistent across games, solvers, configuration, logging, and
experiment workflows. Research findings, chronological progress, and paper notes
belong in the other docs; this file should stay focused on code structure and
architectural invariants.

## Architectural Goals

- Keep game backends interchangeable through a shared interface.
- Keep solver logic independent from whether ranks come from the real Contexto
  API or the local embedding game.
- Keep LLM-based search separate from embedding-neighbor search.
- Keep LLM provider selection isolated from solver logic.
- Preserve traceability: every solver run should produce readable JSON traces
  that explain the search process.
- Support local experiments without rate limits while respecting rate limits
  for the real Contexto API and paid LLM APIs.

## Main Runtime Flow

Single-run commands go through this path:

```text
main.py
  -> contexto_solver.main
    -> config
    -> choose game backend: LocalGame or ContextoAPI
    -> choose solver: SolverLLM or SolverEmbedding
    -> Logger records RUN_CONFIG and solver events
    -> solver.solve()
    -> trace JSON written to traces/
```

Batch local experiments go through this path:

```text
python -m contexto_solver.experiment
  -> load target list
  -> load game embedding model once
  -> optionally load separate solver embedding model
  -> choose LLM provider/model when using the LLM solver
  -> run LocalGame + selected solver repeatedly
  -> write per-run traces plus resumable JSON/CSV summary
```

Manual local play goes through this path:

```text
play.py
  -> contexto_solver.play
    -> EmbeddingModel
    -> LocalGame
    -> interactive terminal loop
```

## Shared Game Contract

All current solvers depend on a small game interface:

```python
guess(word) -> int
total_guesses() -> int
best_so_far() -> tuple[str | None, int | None]
is_solved() -> bool
```

Important invariant: the correct answer is rank `1` for every backend exposed
to solvers. `LocalGame` naturally uses rank `1`; `ContextoAPI` normalizes the
real API's answer distance from `0` to `1`.

When changing either game backend, keep this contract stable. Solver code should
not need backend-specific rank handling.

## Components

### `contexto_solver.main`

Main CLI orchestrator for single runs.

Main function:
- Parses CLI arguments.
- Loads embeddings only when required.
- Constructs either `LocalGame` or `ContextoAPI`.
- Constructs either `SolverLLM` or `SolverEmbedding`.
- Logs `RUN_CONFIG`, including LLM provider/model for LLM runs.
- Prints final status, best word/rank, total guesses, generation count, and
  trace path.

Main interactions:
- Reads defaults from `config`.
- Uses `EmbeddingModel` for local games and embedding solvers.
- Uses `LLMClient` only for LLM solver runs.
- Selects `openai`, `anthropic`, or `ollama` through the existing
  `--provider`/`LLM_PROVIDER` setting.
- Passes a shared `Logger` into the selected solver.

Subtleties:
- `game_embedding_path` and `solver_embedding_path` are separate to support
  aligned and non-aligned embedding experiments.
- If the local game and embedding solver use the same path, the embedding model
  instance is reused to avoid loading GloVe twice.
- `--ollama-model` is a convenience override for Ollama runs; `--model` remains
  the generic model override.
- `_default(value, default)` preserves explicit `0` CLI values; do not replace
  it with `value or default`.

### `contexto_solver.config`

Central configuration module.

Main function:
- Loads simple `.env` key/value pairs into environment variables if absent.
- Defines defaults for paths, named embedding caches, API settings, LLM
  settings, solver budgets, and local-game target.

Main interactions:
- Imported by CLI modules, experiment runner, and manual play.
- Provides path defaults for `EmbeddingModel`, trace output, and embedding
  alignment modes.

Subtleties:
- `.env` values do not overwrite variables already present in the process
  environment.
- `LLM_PROVIDER` is the provider selector for OpenAI, Anthropic, and Ollama.
  Provider-specific defaults include `LLM_MODEL` for cloud providers and
  `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REQUEST_TIMEOUT_SECONDS` for
  local Ollama.
- The default generation budget is `MAX_GENERATIONS=50`; CLI flags can still
  lower it to `0` for smoke tests or raise/lower it per run.
- Named embedding paths include legacy GloVe plus transformer caches for
  MiniLM and MPNet. MiniLM becomes the default local embedding backend once its
  cache exists; otherwise the default falls back to GloVe so a fresh checkout
  still works with the existing validated file.

### `contexto_solver.embeddings.EmbeddingModel`

Embedding loader and nearest-neighbor query engine.

Main function:
- Loads text embeddings or compressed `.npz` embedding caches from disk into
  `numpy` arrays.
- Stores `words`, `vectors`, `norms`, and `word_to_index`.
- Provides vector lookup, vocabulary access, nearest neighbors for a word, and
  nearest neighbors to an arbitrary vector.

Main interactions:
- Used by `LocalGame` to build full-vocabulary rankings.
- Used by `SolverEmbedding` to propose nearest-neighbor guesses.
- Loaded by `main`, `experiment`, and `play`.

Subtleties:
- Loading large embedding matrices is expensive and prints progress for text
  files; avoid loading more than necessary.
- Text files remain compatible with GloVe-style static vectors.
- `.npz` caches must contain `words` and `vectors` arrays. Optional
  `metadata_json` is used for provenance only, not solver behavior.
- Runtime components should depend on this interface, not on transformer model
  APIs directly.

### `contexto_solver.build_embedding_cache`

Static cache builder for transformer embedding models.

Main function:
- Reads a fixed vocabulary file, including GloVe-style embedding files where
  the first whitespace-delimited field is the word.
- Encodes each word with a sentence-transformer model such as
  `sentence-transformers/all-MiniLM-L6-v2` or
  `sentence-transformers/all-mpnet-base-v2`.
- Writes a compressed `.npz` cache compatible with `EmbeddingModel`.

Main interactions:
- Used offline before local-game or embedding-solver runs.
- Produces files under `data/embeddings/` by convention.
- Does not participate in live solving or experiment ranking loops.

Subtleties:
- The vocabulary is fixed at build time. This preserves reproducibility and
  avoids contextual drift between runs.
- Encoded words are treated as static lexical vectors. The local game should
  never call sentence-transformer models dynamically.

### `contexto_solver.local_game.LocalGame`

Offline Contexto-style game backend.

Main function:
- Receives an `EmbeddingModel` and target word.
- Precomputes cosine similarity from the target vector to every vocabulary word.
- Converts similarities into ranks, with target rank `1`.
- Tracks guesses and exposes the shared game interface.

Main interactions:
- Constructed by `main`, `experiment`, and `play`.
- Consumed by both solvers through the shared game contract.

Subtleties:
- Unknown words return `-1` and are still stored in `guesses`.
- `best_so_far()` ignores invalid ranks (`<= 0`).
- Local games have no rate limit.

### `contexto_solver.game_api.ContextoAPI`

Current real Contexto API backend.

Main function:
- Sends guesses to the public Contexto endpoint.
- Applies a configurable delay before network requests.
- Caches valid guesses and invalid guesses.
- Normalizes API answer distance `0` to shared rank `1`.
- Returns `-1` for unavailable/invalid ranks.

Main interactions:
- Constructed by `main` for `--game api`.
- Consumed by both solvers through the shared game contract.

Subtleties:
- The rate limit belongs here, not in solver code. Local games should remain
  delay-free.
- Bad HTTP responses, request failures, and malformed API responses are treated
  as invalid guesses so solvers can continue.
- Do not reintroduce backend-specific solved-rank logic into solvers.

### `contexto_solver.llm_client.LLMClient`

LLM provider wrapper and prompt owner.

Main function:
- Supports OpenAI, Anthropic, and Ollama chat APIs.
- Owns prompt templates for initial categories, word proposals, mutation,
  crossover, local search, and pivot operators.
- Requests JSON-only responses and parses JSON. JSON parse failures retry the
  generation request; retryable provider failures use bounded backoff.

Main interactions:
- Used only by `SolverLLM`.
- Receives `Hypothesis` state and rank feedback from the solver.
- Uses invalid/global guess sets supplied by `SolverLLM` to reduce repeated or
  unusable words.
- Receives the selected provider, API key placeholder/key, and resolved model
  from `main` or `experiment`.

Subtleties:
- Prompts enforce single lowercase words because Contexto rejects phrases,
  punctuation, and hyphenated guesses.
- `local_search()` is LLM-based; it does not use the local game's embedding
  model. This preserves separation between LLM solver and game internals.
- OpenAI and Anthropic use raw `requests` calls to their native APIs. Ollama
  uses raw `requests` against its OpenAI-compatible
  `{OLLAMA_BASE_URL}/chat/completions` endpoint; no extra Ollama SDK is used.
- Ollama does not require a real API key. If the local server is unreachable or
  the selected model is missing, `LLMClient` raises a clear error instead of
  falling back to a cloud provider.
- Provider errors are raised by `requests` after response status checks unless
  they are converted into clearer local Ollama errors first.

### `contexto_solver.hypothesis.Hypothesis`

State model for one LLM search category.

Main function:
- Stores category name, description, tried words and ranks, activity status,
  parent, and origin.
- Computes best rank and best word for that category.
- Serializes itself for traces.

Main interactions:
- Created and mutated by `SolverLLM`.
- Passed into `LLMClient` for prompt context.
- Logged through `Logger` as part of initialization, selection, mutation,
  crossover, and deduplication traces.

Subtleties:
- Empty hypotheses use a large sentinel best rank.
- `parent` and `origin` are traceability fields; preserve them when adding new
  hypothesis-generation operations.

### `contexto_solver.solver_llm.SolverLLM`

LLM-guided evolutionary solver.

Main function:
- Initializes broad semantic hypotheses from the LLM.
- In each generation:
  - caps active hypotheses,
  - asks the LLM for candidate words,
  - evaluates candidates against the game,
  - optionally performs LLM local search near strong clues,
  - selects active hypotheses with elitism,
  - mutates strong hypotheses,
  - performs crossover,
  - deduplicates similar hypotheses,
  - detects stalls and can run pivot operators.
- Prints best word/rank after generation `0` and each completed generation.
- Saves a JSON trace at the end.

Main interactions:
- Depends on the shared game interface, `LLMClient`, `Hypothesis`, and `Logger`.
- Does not directly depend on `EmbeddingModel`, `LocalGame`, or `ContextoAPI`.

Subtleties:
- This solver must remain backend-agnostic. It should use only ranks returned by
  the game interface, not embedding vectors or local target internals.
- Invalid guesses return `-1` and are remembered to avoid repeated proposals.
- Candidate generation uses global guess history when available to reduce
  duplicates across hypotheses.
- Local search is triggered by `local_search_rank_threshold`; it is not a
  replacement for category exploration in all cases.
- Pivoting is still LLM-based. It adds morphology, register-shift, and adjacent
  category directions when the stall detector fires.
- The active hypothesis cap and deduplication are important performance
  safeguards. Changes to selection/mutation should consider their impact on
  diversity and convergence.

### `contexto_solver.solver_embedding.SolverEmbedding`

Embedding-neighbor evolutionary baseline.

Main function:
- Samples initial seed words from the solver embedding vocabulary.
- Keeps the best active words by rank.
- In each generation, proposes nearest neighbors of active words and wider
  neighbors around the current best word.
- Prints best word/rank after generation `0` and each completed generation.
- Saves a JSON trace at the end.

Main interactions:
- Depends on the shared game interface, `EmbeddingModel`, and `Logger`.
- Can run against `LocalGame` or `ContextoAPI`.

Subtleties:
- When solver and game embedding paths match, this is an aligned condition.
- When paths differ, it is a non-aligned condition.
- Current non-aligned support is architectural scaffolding; only GloVe has been
  validated so far.
- This solver can know its own solver embedding model, but it should not assume
  the game backend uses the same model unless explicitly configured that way.

### `contexto_solver.logger.Logger`

In-memory trace builder and JSON writer.

Main function:
- Appends structured trace entries with generation, event, timestamp, and
  details.
- Writes full traces as indented JSON files.
- Can print final summary from `SOLVED` or `FAILED` events.

Main interactions:
- One `Logger` is created per run by `main` or `experiment`.
- Solvers write all major events through it.
- Trace files are saved by solvers using their configured `trace_dir` and
  `run_label`.

Subtleties:
- The logger keeps traces in memory until the run ends.
- The logger does not enforce event schemas; consistency is the responsibility
  of solvers and orchestrators.

### `contexto_solver.experiment`

Batch local experiment runner.

Main function:
- Runs repeated local experiments over targets from CLI or target file.
- Supports `llm` and `embedding` solvers.
- Supports `aligned` and `non_aligned` embedding modes.
- Writes per-run traces plus aggregate JSON and CSV summaries.
- Can resume an existing summary with `--resume`, skipping target/run pairs
  already present in the output.

Main interactions:
- Always uses `LocalGame`.
- Reuses `EmbeddingModel` instances when possible.
- Constructs `SolverLLM` or `SolverEmbedding` similarly to `main`.
- Records LLM provider/model in experiment metadata, per-run rows, CSV output,
  and each run's `RUN_CONFIG` trace event for LLM experiments.

Subtleties:
- Real API batch experiments are intentionally not handled here.
- Alignment validation is enforced for embedding solver runs.
- `random_seed` is offset by run index for repeated embedding experiments.
- Summary files are rewritten after each completed run so interrupted batches
  can be resumed without losing completed results.

### `contexto_solver.play`

Manual local-game terminal interface.

Main function:
- Loads embeddings.
- Creates a `LocalGame`.
- Accepts typed guesses.
- Prints rank, best-so-far, hints, and final target.

Main interactions:
- Uses `config`, `EmbeddingModel`, and `LocalGame`.
- Does not use solvers or `Logger`.

Subtleties:
- This is a manual validation/debug tool for the local game.
- It should stay lightweight and separate from automatic solver logic.

### Root Wrappers: `main.py` and `play.py`

Convenience scripts at repository root.

Main function:
- `main.py` calls `contexto_solver.main.main`.
- `play.py` calls `contexto_solver.play.main`.

Main interactions:
- They exist for CLI ergonomics only.

Subtleties:
- Avoid adding application logic here; keep logic in package modules.

## Data And Output Directories

### `data/`

Local embedding files live here and are not committed.

Current assumption:
- `data/glove.6B.300d.txt` remains the legacy validated embedding file.
- `data/embeddings/all-MiniLM-L6-v2.npz` is the intended default local-game
  backend after cache generation.
- `data/embeddings/all-mpnet-base-v2.npz` is the heavier quality-oriented
  solver/backend option after cache generation.

### `traces/`

Generated solver traces and experiment summaries live here.

Current outputs:
- Per-run JSON traces from solvers.
- Batch experiment JSON summaries.
- Batch experiment CSV summaries.
- LLM experiment outputs include `llm_provider` and `llm_model` so local and
  cloud model results are distinguishable during analysis.

Subtleties:
- Trace files are evidence for experiments, but they can be large.
- Generated traces should not be treated as source modules.

### `docs/`

Project documentation and research notes live here.

Current important docs:
- `docs/architecture.md`: this architecture reference.
- `docs/design_decisions.md`: algorithmic and experimental design rationale.
- `docs/experiment_log.md`: compact experiment register and evidence index.
- `docs/findings.md`: paper-facing findings and evidence-quality notes.
- `docs/research_timeline.md`: chronological project timeline.

## Legacy Modules

### `contexto_solver.contexto_api`

Older Contexto API wrapper.

Status:
- Not used by the current `contexto_solver.main` path.
- Superseded by `contexto_solver.game_api`, which implements the shared game
  contract and rank normalization.

Guidance:
- Do not add new behavior here unless intentionally migrating or deleting the
  legacy path.

### `contexto_solver.solver`

Older LLM solver implementation.

Status:
- Not used by the current `contexto_solver.main` path.
- Superseded by `contexto_solver.solver_llm`, which uses the shared game
  interface and newer performance mitigations.

Guidance:
- Future LLM solver changes should target `solver_llm.py`, not this legacy
  module, unless the project explicitly decides to remove or revive it.

## Change-Impact Checklist

Before modifying a component, check these likely dependents:

- Game rank behavior: update/verify `LocalGame`, `ContextoAPI`, both solvers,
  and trace interpretation.
- Game interface shape: update `SolverLLM`, `SolverEmbedding`, `main`,
  `experiment`, and `play` if needed.
- Embedding loading or paths: update `config`, `main`, `experiment`,
  `EmbeddingModel`, `LocalGame`, `SolverEmbedding`, README, and docs.
- LLM prompts or JSON schemas: update `LLMClient`, `SolverLLM` parsing/cleaning,
  trace expectations, and experiment notes.
- LLM provider routing or defaults: update `config`, `main`, `experiment`,
  `LLMClient`, `RUN_CONFIG` metadata, experiment summary fields, and smoke
  tests for both selected provider behavior and provider-specific errors.
- Selection, mutation, local search, or deduplication: update `SolverLLM`,
  verify traces, and consider effects on convergence/diversity.
- Experiment summary fields: update `experiment`, CSV fieldnames, downstream
  docs, and any scripts that read summaries.
- Trace event names/details: update solvers, `Logger` consumers, documentation,
  and any manual analysis assumptions.

## Preserved Invariants

- Solvers see rank `1` as solved regardless of backend.
- Invalid/unavailable guesses use `-1` at the game interface.
- Local games should not use API rate limiting.
- LLM solver should not access local embedding vectors.
- LLM provider choice should not change solver, hypothesis, game, or trace event
  interfaces beyond run-level provider/model metadata.
- A selected LLM provider should fail clearly on provider-specific errors; do
  not silently fall back to a different provider.
- Embedding solver should only use its configured solver embedding model.
- Root wrappers should remain thin.
- Generated traces and experiment summaries should not become required inputs
  for normal single-run solving.
