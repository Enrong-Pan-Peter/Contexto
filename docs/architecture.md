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
    -> choose method: llm_only, ea_llm, ea_llm_pivot, ea_llm_self_adaptive, or embedding
    -> Logger records RUN_CONFIG and solver events
    -> method.solve()
    -> trace JSON written to traces/
```

Batch local experiments go through this path:

```text
python -m contexto_solver.experiment
  -> load target list
  -> load game embedding model once
  -> optionally load separate solver embedding model
  -> choose LLM provider/model when using an LLM method
  -> run LocalGame + selected method repeatedly
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
- Constructs the selected method from `contexto_solver.methods`.
- Logs `RUN_CONFIG`, including LLM provider/model for LLM runs.
- Prints final status, best word/rank, total guesses, generation count, and
  trace path.

Main interactions:
- Reads defaults from `config`.
- Uses `EmbeddingModel` for local games and embedding solvers.
- Uses `LLMClient` only for LLM-family method runs.
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
- `--method` is the method selector. The `solver` metadata field is retained as
  a compatibility family value (`llm` or `embedding`) so existing traces and
  analysis scripts can continue to group runs at a coarse level. In particular,
  `solver=llm` can now mean `method=llm_only`, `method=ea_llm`,
  `method=ea_llm_pivot`, or `method=ea_llm_self_adaptive`.
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
- Pivot-only settings are namespaced with `EA_LLM_PIVOT_*` and only apply to
  `method=ea_llm_pivot`. There is no global `ENABLE_PIVOT` flag; pivot behavior
  is selected by method.

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
- Owns prompt templates for initial categories, pure LLM next guesses, word
  proposals, mutation, crossover, local search, and pivot operators.
- Requests JSON-only responses and parses JSON. JSON parse failures retry the
  generation request; retryable provider failures use bounded backoff.

Main interactions:
- Used only by LLM-family methods.
- Receives `Hypothesis` state and rank feedback from EA methods.
- Uses invalid/global guess sets supplied by methods to reduce repeated or
  unusable words.
- Receives the selected provider, API key placeholder/key, and resolved model
  from `main` or `experiment`.

Subtleties:
- Prompts enforce single lowercase words because Contexto rejects phrases,
  punctuation, and hyphenated guesses.
- `next_guess()` supports the pure `llm_only` baseline by conditioning on the
  running `(word, rank)` history.
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
  parent, origin, stable hypothesis ID, canonical parent ID, and self-adaptive
  sigma.
- Computes best rank and best word for that category.
- Serializes itself for traces.

Main interactions:
- Created and mutated by `SolverLLM`.
- Passed into `LLMClient` for prompt context.
- Logged through `Logger` as part of initialization, selection, mutation,
  crossover, and deduplication traces.

Subtleties:
- Empty hypotheses use a large sentinel best rank.
- `parent` and `origin` are compatibility traceability fields; preserve them
  when adding new hypothesis-generation operations.
- `parent_id` is the canonical lineage identifier for logging and analysis.
  When a child is created from a parent hypothesis, `parent_id` should be set to
  the parent's `hypothesis_id`.
- `sigma` is serialized for self-adaptive analysis. It must always have four
  components, sum to one within `1e-6`, and remain above the configured
  self-adaptive floor for generated adaptive descendants.

### `contexto_solver.operators`

Shared self-adaptive mutation operator definitions.

Main function:
- Defines exactly four operator IDs: `s_mutation`, `m_mutation`,
  `ml_mutation`, and `l_mutation`.
- Provides uniform initial sigma vectors, categorical operator sampling, and
  Dirichlet perturbation with a configurable floor.
- Maps operator IDs to prompt constants owned by `contexto_solver.llm_client`.
- Provides a prompt-leakage assertion used by the self-adaptive method and tests
  to ensure sigma values are not included in mutation prompts.

Main interactions:
- Imported by `methods/ea_llm_self_adaptive.py` for mutation sampling.
- Imports prompt constants from `llm_client`; prompt text is not duplicated in
  the operator module.
- Uses `numpy` arrays for sigma vectors.

Subtleties:
- Sigma vectors must have shape `(4,)`, sum to `1.0` within `1e-6`, and
  generated adaptive descendants should satisfy `sigma.min() >=
  SELF_ADAPTIVE_SIGMA_FLOOR`.
- The LLM receives only the selected operator's formatted prompt. It never sees
  sigma values, operator probabilities, or the list of alternatives.
- Operator IDs are trace-facing identifiers through `OPERATOR_SAMPLED` events.

### `contexto_solver.methods`

Method package for automatic solvers.

Main function:
- Hosts one module per experimental method.
- Gives each method a `.solve(max_generations=None) -> dict` interface.
- Keeps method implementations separate from game backend construction and CLI
  dispatch.

Current method modules:
- `methods/llm_only.py`: pure LLM baseline. It asks the LLM for one
  history-conditioned guess per step and uses no hypotheses, mutation,
  crossover, or local search.
- `methods/ea_core.py`: shared EA+LLM core for hypothesis initialization,
  candidate generation, local search, selection, mutation, crossover,
  deduplication, trace saving, and the post-generation hook contract.
- `methods/ea_llm.py`: EA+LLM without stall-pivot operators.
- `methods/ea_llm_pivot.py`: EA+LLM with the stall detector and pivot A/B/C
  operators. Pivot settings are read from `EA_LLM_PIVOT_*` config values.
- `methods/ea_llm_self_adaptive.py`: EA+LLM variant whose mutation and
  crossover steps carry per-hypothesis operator probability vectors and emit
  sigma telemetry.
- `methods/embedding.py`: embedding nearest-neighbor baseline.
- `methods/base.py`: shared `Game` and `SolverMethod` protocols.

Main interactions:
- `main` and `experiment` select a method with `--method`.
- LLM-family methods use `LLMClient` and keep `solver=llm` compatibility
  metadata.
- The embedding method uses its configured `EmbeddingModel` and keeps
  `solver=embedding` compatibility metadata.

Subtleties:
- `BaseEALLMMethod._after_generation_update()` is a method-specific hook. Hooks
  may mutate shared solver state, log events, or update trackers; its return
  value only means whether the hook solved the game and the loop should stop.
- `ea_llm_pivot` is a subclass of the EA core because pivoting is attached at
  one post-generation hook point but needs access to EA internals such as
  hypotheses, known guesses, and guess/update logging.
- `ea_llm_self_adaptive` is also built on the EA core but overrides mutation,
  crossover, local search, and active-population capping. Mutation samples one
  of the four backend operators from the parent's sigma, crossover perturbs the
  average of the two parent sigma vectors, and local search is disabled by
  default through the method config so adaptive runs do not inject uniform-sigma
  local-search hypotheses. Selection, candidate generation, and deduplication
  remain on the shared path.
- `solver=llm` is no longer a unique method. Analysis scripts should inspect
  `method` when distinguishing `llm_only`, `ea_llm`, `ea_llm_pivot`, and
  `ea_llm_self_adaptive`.
- The `enable_pivot` metadata field remains only as a compatibility field for
  pivot matrix analysis: `ea_llm` and `ea_llm_self_adaptive` write `False`,
  `ea_llm_pivot` writes `True`, and other methods write `None`.
- All methods must remain backend-agnostic. LLM methods use only rank feedback,
  not local embedding vectors or target internals.
- The embedding method can know its own solver embedding model, but it should
  not assume the game backend uses the same model unless explicitly configured.

### `contexto_solver.logger.Logger`

In-memory trace builder and JSON writer.

Main function:
- Appends structured trace entries with generation, event, timestamp, and
  details.
- Writes full traces as indented JSON files.
- Can print final summary from `SOLVED` or `FAILED` events.
- Provides convenience methods for self-adaptive `OPERATOR_SAMPLED` and
  `SIGMA_TRAJECTORY` events.

Main interactions:
- One `Logger` is created per run by `main` or `experiment`.
- Solvers write all major events through it.
- Trace files are saved by solvers using their configured `trace_dir` and
  `run_label`.

Subtleties:
- The logger keeps traces in memory until the run ends.
- The logger does not enforce event schemas; consistency is the responsibility
  of solvers and orchestrators.
- Self-adaptive traces use `OPERATOR_SAMPLED` for each sampled child mutation
  and `SIGMA_TRAJECTORY` for generation-level population means.
  `OPERATOR_SAMPLED.details` includes `parent_id`, `child_id`,
  `sigma_snapshot`, `child_sigma`, `sampled_op`, and `method`. Self-adaptive
  `CROSSOVER` details extend the shared child record with `parent_a_sigma`,
  `parent_b_sigma`, and `child_sigma_pre_perturbation`; the post-perturbation
  child sigma is serialized in `child.sigma`. When adaptive local search is
  muted, `LOCAL_SEARCH_DISABLED` may be logged once to make the suppression
  visible in traces.

### `contexto_solver.experiment`

Batch local experiment runner.

Main function:
- Runs repeated local experiments over targets from CLI or target file.
- Supports `llm_only`, `ea_llm`, `ea_llm_pivot`, and `embedding` methods.
- Supports `aligned` and `non_aligned` embedding modes.
- Writes per-run traces plus aggregate JSON and CSV summaries.
- Can resume an existing summary with `--resume`, skipping target/run pairs
  already present in the output.

Main interactions:
- Always uses `LocalGame`.
- Reuses `EmbeddingModel` instances when possible.
- Constructs the selected method similarly to `main`.
- Records LLM provider/model in experiment metadata, per-run rows, CSV output,
  and each run's `RUN_CONFIG` trace event for LLM experiments.

Subtleties:
- Real API batch experiments are intentionally not handled here.
- Alignment validation is enforced for embedding method runs.
- `random_seed` is offset by run index for repeated embedding experiments.
- Summary files are rewritten after each completed run so interrupted batches
  can be resumed without losing completed results.

### `contexto_solver.plot_trajectory`

Standalone trace-visualization and trajectory-analysis script.

Main function:
- Reads existing solver trace JSON files from `traces/`.
- Loads embeddings through `EmbeddingModel` only when a plot needs geometry.
- Computes target-neighborhood explained variance with PCA.
- Projects single-run search trajectories into 2D with PCA, UMAP, or PaCMAP.
- Plots rank trajectories and cosine-distance-to-target trajectories directly
  from traces.
- Writes figures to `figures/` by default, or to a user-specified `--output`.

Main interactions:
- Consumes generated traces; it does not participate in solving or batch
  execution.
- Uses `config.GAME_EMBEDDING_PATH` as the default embedding path for
  embedding-dependent analyses.
- Uses `EmbeddingModel` for GloVe text files and `.npz` embedding caches.
- Uses trace event fields such as `RUN_CONFIG.details.target`, `GUESS`,
  per-event `best_word`/`best_rank`, and serialized hypotheses to reconstruct
  trajectories.

Subtleties:
- `--plot-type single` is the default CLI mode. Additional modes are `multi`,
  `rank`, `distance`, and `variance`; the older `--variance` shortcut remains
  supported.
- `rank` plots need only a trace. `distance`, `single`, `multi`, and projection
  variance checks need an embedding model matching the trace's local-game
  embedding when interpreting geometric paths.
- UMAP and PaCMAP are fitted on the target neighborhood and then all projected
  words are transformed through a shared per-word coordinate cache. This keeps
  repeated words, such as a target that is later guessed, at identical
  coordinates.
- Active-hypothesis centroid, rank, and distance trajectories are reconstructed
  from trace events. Because traces store category names rather than stable
  hypothesis IDs, duplicate category names produce a warning and make those
  reconstructed population-level lines approximate.
- Plotting dependencies are intentionally local to the analysis module; solver
  code should not import Matplotlib, UMAP, PaCMAP, or scikit-learn projection
  APIs.

### `scripts/inspect_self_adaptive_trace.py`

Standalone self-adaptive trace inspection script.

Main function:
- Reads an existing self-adaptive trace JSON file and writes inspection figures
  next to that trace.
- Performs five checks: perturbation magnitude, parent-id integrity, mean sigma
  drift, best-lineage sigma trajectory, and operator usage.
- Produces `mean_sigma_over_generations.png`,
  `best_lineage_sigma_trajectory.png`, and `operator_usage_histogram.png` as
  analysis outputs.

Main interactions:
- Consumes trace events such as `OPERATOR_SAMPLED`, `SIGMA_TRAJECTORY`,
  serialized hypothesis records, and self-adaptive `CROSSOVER` child records.
- Does not participate in solving or modify traces.

Subtleties:
- Legacy traces that predate full mutation-child records or crossover sigma
  blending can be partially inspectable but may not support every lineage check.
- Inspection output is evidence about a specific trace, not a batch-level result.

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

### `figures/`

Generated visualization outputs live here.

Current outputs:
- Single-run trajectory plots from `contexto_solver.plot_trajectory`.
- Rank and distance trajectory plots for trace inspection.
- Projection comparison figures for PCA, UMAP, or PaCMAP views.

Subtleties:
- Figures are analysis artifacts, not runtime inputs.
- They should be cited as evidence for qualitative inspection only unless tied
  to repeated-run summaries or batch-level statistics.

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
- Superseded by `contexto_solver.methods`, which uses the shared game interface
  and method-based dispatch.

Guidance:
- Future LLM method changes should target `contexto_solver.methods`, not this
  legacy module, unless the project explicitly decides to remove or revive it.

## Change-Impact Checklist

Before modifying a component, check these likely dependents:

- Game rank behavior: update/verify `LocalGame`, `ContextoAPI`, both solvers,
  and trace interpretation.
- Game interface shape: update methods, `main`, `experiment`, and `play` if
  needed.
- Embedding loading or paths: update `config`, `main`, `experiment`,
  `EmbeddingModel`, `LocalGame`, `methods/embedding.py`, README, and docs.
- LLM prompts or JSON schemas: update `LLMClient`, LLM methods'
  parsing/cleaning, trace expectations, and experiment notes.
- LLM provider routing or defaults: update `config`, `main`, `experiment`,
  `LLMClient`, `RUN_CONFIG` metadata, experiment summary fields, and smoke
  tests for both selected provider behavior and provider-specific errors.
- Selection, mutation, local search, or deduplication: update
  `methods/ea_core.py`, verify traces, and consider effects on
  convergence/diversity.
- Self-adaptive operators or sigma behavior: update `operators.py`,
  `hypothesis.py`, `methods/ea_llm_self_adaptive.py`, prompt-leakage tests, and
  trace inspection scripts.
- Experiment summary fields: update `experiment`, CSV fieldnames, downstream
  docs, and any scripts that read summaries.
- Trace event names/details: update solvers, `Logger` consumers, documentation,
  and any manual analysis assumptions.
- Analysis visualizations: update `contexto_solver.plot_trajectory`,
  `requirements.txt`, and docs when trace interpretation, projection methods, or
  generated figure conventions change.

## Preserved Invariants

- Solvers see rank `1` as solved regardless of backend.
- Invalid/unavailable guesses use `-1` at the game interface.
- Local games should not use API rate limiting.
- LLM methods should not access local embedding vectors.
- Self-adaptive sigma must remain backend-only metadata; it should be logged for
  analysis but never included in LLM prompts.
- LLM provider choice should not change solver, hypothesis, game, or trace event
  interfaces beyond run-level provider/model metadata.
- A selected LLM provider should fail clearly on provider-specific errors; do
  not silently fall back to a different provider.
- Embedding method should only use its configured solver embedding model.
- Root wrappers should remain thin.
- Generated traces and experiment summaries should not become required inputs
  for normal single-run solving.
- Analysis scripts may read traces and embeddings, but they must not modify
  solver behavior, game backends, or trace schemas.
