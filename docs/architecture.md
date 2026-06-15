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
    -> choose method: llm_only, ea_llm, ea_llm_pivot, ea_llm_self_adaptive, ea_llm_map_elites, or embedding
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
- Self-adaptive settings are namespaced with `SELF_ADAPTIVE_*`.
  `SELF_ADAPTIVE_INITIAL_CATEGORIES` controls only the number of initial
  categories requested by `method=ea_llm_self_adaptive`, and
  `SELF_ADAPTIVE_MU` controls only that method's active population cap and
  mutation-parent count. Regular `ea_llm` and `ea_llm_pivot` continue to use
  `INITIAL_CATEGORIES` and `MAX_ACTIVE_HYPOTHESES`.
- MAP-Elites settings are namespaced with `MAPELITES_*` and read only by
  `method=ea_llm_map_elites`. It uses `MAPELITES_INITIAL_CATEGORIES`,
  `MAPELITES_GRID_RESOLUTION`, `MAPELITES_MUTATIONS_PER_GEN`,
  `MAPELITES_CROSSOVERS_PER_GEN`, `MAPELITES_PLACEMENT_CACHE_DIR`, and the
  `MAPELITES_ANCHORS_*` scales, while reusing the `SELF_ADAPTIVE_*`
  concentration and sigma-floor values for the inherited operator adaptation.
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
- The four operator mutation templates carry a shared `{ranked_context}` slot
  (right after the best-word/rank line). `build_operator_mutation_prompt` only
  fills it when the template contains the slot, defaulting to the empty string,
  so self-adaptive prompts stay byte-identical. Only MAP-Elites passes a
  non-empty `ranked_context` (top-K best-ranked guessed words). Like the
  `{all_guesses}` slot, the injected vocabulary is not reserved-substring
  filtered; the values are game ranks, never sigma, so the leakage invariant
  asserted by `assert_prompt_has_no_sigma_leak` holds.

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
- Child sigma is drawn from `Dirichlet(alpha * parent_sigma)` plus floor
  renormalization. It depends only on the parent's sigma, not on which operator
  was sampled; the sampled operator determines the child's word-generation
  prompt, not the inherited sigma vector.
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
- `methods/ea_llm_map_elites.py`: MAP-Elites variant of the self-adaptive
  method. Replaces top-mu selection with a behavior archive (see the dedicated
  subsection below).
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
- `ea_llm_map_elites` subclasses `ea_llm_self_adaptive` and overrides
  `initialize()` and `run_generation()` whole-cloth. Because the base EA loop is
  not invoked, per-hypothesis multi-candidate generation, top-mu/half selection,
  the post-generation hook, and deduplication are all inactive in this method.
  It inherits the sigma machinery unchanged (`_mutate`/`_crossover` math reused
  via per-child helpers) and keeps the four-operator sigma vectors. See the
  dedicated subsection below.
- `solver=llm` is no longer a unique method. Analysis scripts should inspect
  `method` when distinguishing `llm_only`, `ea_llm`, `ea_llm_pivot`,
  `ea_llm_self_adaptive`, and `ea_llm_map_elites`.
- The `enable_pivot` metadata field remains only as a compatibility field for
  pivot matrix analysis: `ea_llm`, `ea_llm_self_adaptive`, and
  `ea_llm_map_elites` write `False`, `ea_llm_pivot` writes `True`, and other
  methods write `None`.
- All methods must remain backend-agnostic. LLM methods use only rank feedback,
  not local embedding vectors or target internals.
- The embedding method can know its own solver embedding model, but it should
  not assume the game backend uses the same model unless explicitly configured.

### `methods/ea_llm_map_elites.py`

`ea_llm_map_elites` is a MAP-Elites quality-diversity layer on top of the
self-adaptive operators. It keeps the inherited sigma self-adaptation but
replaces selection with a behavior archive.

Archive:
- A `grid_resolution x grid_resolution` grid (default `5x5 = 25` cells), stored
  as `self.archive: dict[(i, j), Hypothesis]` holding only occupied cells. Each
  cell holds zero or one elite. The "population" is the number of occupied cells
  (0 to 25).
- A hypothesis's behavior coordinate is the coordinate of its single
  `best_word`, fixed at creation. Hypotheses carry optional
  `coordinates: (float, float)` and `cell: (int, int)` fields (default `None`,
  so other methods are unaffected) that are serialized in the trace.

Behavior axes (anchored, LLM-placed):
- Concreteness: `0` = most concrete/physical, `1` = most abstract/conceptual.
- Specificity: `0` = most general, `1` = most specific.
- Anchor words and scale positions come from `MAPELITES_ANCHORS_CONCRETENESS`
  and `MAPELITES_ANCHORS_SPECIFICITY`, so anchors are tunable without code
  changes. Placement is a single LLM call (`LLMClient.place_word`) returning
  `{concreteness, specificity}` in `[0, 1]`. The cell is
  `clip(floor(coord * resolution), 0, resolution - 1)` per axis. No embedding
  centroid math is used, so the method works against the real Contexto API.

Placement cache:
- Results are cached to `MAPELITES_PLACEMENT_CACHE_DIR/{model}_{anchors_hash}.json`
  with schema `{word: [concreteness, specificity]}`. `anchors_hash` is a stable
  hash over the sorted `(axis, position, word)` anchor triples, so any anchor
  change automatically invalidates stale entries. The cache loads on init and
  persists on each new placement (durability over speed). `cache_hit` is logged
  on every `PLACEMENT` event.

Per-generation loop (`run_generation`):
- Sample `MAPELITES_MUTATIONS_PER_GEN` mutation parents uniformly with
  replacement over occupied cells; each yields one sigma-driven mutation child
  with exactly one `best_word` (operator prompt requested with `n=1`).
- Sample `MAPELITES_CROSSOVERS_PER_GEN` crossover pairs (two parents each, with
  replacement); each yields one blended-sigma crossover child with one
  `best_word`.
- For each child: guess its candidate word (becomes `best_word`), LLM-place it,
  compute its cell, then apply per-cell competition: empty cell -> place;
  otherwise the better-ranked hypothesis stays. A fresh-jump child survives if
  its cell is empty or its incumbent is worse, which is the selection-layer fix
  for the diversity problem.
- Pipeline invariant: MAP-Elites uses exactly one proposed word per child. If
  that word is already known, `_guess_first_valid()` drops the child before any
  archive logic and no `GUESS`, `PLACEMENT`, archive event, `OPERATOR_SAMPLED`,
  or `CROSSOVER` record is emitted for that attempt. If the word is invalid, it
  may emit `SKIP_INVALID_GUESS`, but it still produces no archive competition.
  Therefore successful children correspond one-to-one with valid `GUESS`
  events, `PLACEMENT` events, and archive outcomes.
- `_active_hypotheses()` is overridden to return the current archive incumbents
  so the inherited sigma-trajectory logging reflects the archive.

Trace events emitted by this method (in addition to inherited ones):
- `AXIS_DEFINITION` (once at run start): anchors, scale positions, grid
  resolution, so placements are re-derivable from the trace.
- `PLACEMENT` (per placement): `word`, `coordinates`, `cell`, `cache_hit`.
- `ARCHIVE_PLACE`: new hypothesis into an empty cell.
- `ARCHIVE_REPLACE`: incumbent replaced by a better child.
- `ARCHIVE_REJECT`: child loses to incumbent.
- `ARCHIVE_SNAPSHOT` (end of generation): occupied cells with incumbents'
  `best_word`, rank, coordinates, and sigma vectors.

Config (namespaced `MAPELITES_*`): `MAPELITES_GRID_RESOLUTION`,
`MAPELITES_MUTATIONS_PER_GEN`, `MAPELITES_CROSSOVERS_PER_GEN`,
`MAPELITES_INITIAL_CATEGORIES`, `MAPELITES_PLACEMENT_CACHE_DIR`,
`MAPELITES_ANCHORS_CONCRETENESS`, `MAPELITES_ANCHORS_SPECIFICITY`,
`MAPELITES_SIGMA_MODE`, `MAPELITES_FROZEN_SIGMA`, `MAPELITES_RANKED_CONTEXT_K`.
Sigma behavior reuses the `SELF_ADAPTIVE_*` concentration and floor values.

Sigma-mode control (`MAPELITES_SIGMA_MODE`, env-overridable): selects how each
child's operator-probability vector (sigma) is assigned. Applied at every
creation site (`initialize()`, `_mutation_child`, `_crossover_child`) via the
`_mode_sigma` helper, so frozen/random arms never start generation 0 from an
adaptive parent. `sample_operator(parent.sigma, rng)` is unchanged, so the
operator that actually fires for a child still follows that child's parent
sigma; the mode only controls what sigma children inherit.
- `adaptive` (default, current behavior): Dirichlet perturbation of the parent
  (or blended) sigma using the `SELF_ADAPTIVE_*` concentration and floor.
- `frozen_uniform`: every child sigma is the uniform prior (`initial_sigma()`).
- `frozen_fixed`: every child sigma is `MAPELITES_FROZEN_SIGMA` (order
  `[s, m, ml, l]`, default `[0.4, 0.3, 0.2, 0.1]`, validated to a simplex).
- `random`: every child sigma is a fresh `Dirichlet(1)` draw.

Ranked context (`MAPELITES_RANKED_CONTEXT_K`, default `0` = off): when `> 0`,
mutation prompts gain a line listing the global top-K best-ranked guessed words
(`Closest words found so far: word (rank), ...`). This is a shared
`{ranked_context}` prompt slot (see `LLMClient`); only MAP-Elites populates it,
via `_render_ranked_context()` over a `_word_ranks` tracker maintained in
`_guess_first_valid`. These are game ranks (feedback), not sigma, so the
sigma-leakage invariant is preserved.

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
- Supports `llm_only`, `ea_llm`, `ea_llm_pivot`,
  `ea_llm_self_adaptive`, `ea_llm_map_elites`, and `embedding` methods.
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
- For `ea_llm_map_elites`, experiment metadata and `RUN_CONFIG` also record
  `mapelites_sigma_mode`, `mapelites_frozen_sigma`, and
  `mapelites_ranked_context_k`. Per-run rows and the CSV add
  `mapelites_sigma_mode`, `mapelites_ranked_context_k`, and
  `final_archive_sigma_{s,m,ml,l}` (mean per-operator sigma over the final
  archive incumbents, or `None` for non-MAP-Elites or failed runs).

Subtleties:
- Real API batch experiments are intentionally not handled here.
- Alignment validation is enforced for embedding method runs.
- `random_seed` is offset by run index for repeated embedding experiments.
- Summary files are rewritten after each completed run so interrupted batches
  can be resumed without losing completed results.

### `scripts/run_sigma_control.py`

Batch orchestrator for the MAP-Elites sigma-mode control.

Main function:
- For each selected sigma mode (arm), launches one
  `python -m contexto_solver.experiment --method ea_llm_map_elites` subprocess
  with `MAPELITES_SIGMA_MODE` (and a constant `MAPELITES_RANKED_CONTEXT_K`) set
  in the child environment, writing `traces/sigma_control_<mode>.json`.
- After each arm succeeds, runs `python -m contexto_solver.plot_map_elites
  --trace ...` for every trace listed in that arm's summary.

Main interactions:
- Reuses the existing `experiment` and `plot_map_elites` entrypoints rather than
  reimplementing solving or plotting; it only sets per-arm environment overrides.
- Holds anchors, ranked-context K, targets, seeds, and generations constant
  across arms so the only varying factor is the sigma mode.

Subtleties:
- `--dry-run` prints the per-arm environment overrides and commands without
  executing, for inspection on an HPC node before committing compute.
- A failed experiment or plot for one arm/trace is reported and skipped; it does
  not abort the remaining arms.

### `scripts/compare_sigma_control_arms.py`

Arm-comparison analysis for the MAP-Elites sigma-mode control. It is the
companion to `scripts/run_sigma_control.py` (which produces the batch) and is
distinct from `scripts/measure_sigma_fitness_coupling.py` (which *pools* runs to
estimate an operator-fitness gradient and does not separate the arms).

Main function:
- Loads sigma-control runs, groups them by `MAPELITES_SIGMA_MODE`, and reports,
  paired by `(target, seed)`: per-arm `best_rank` distribution, solve rate,
  archive occupancy, and the final per-operator archive sigma (order
  `[s, m, ml, l]`) read from the last `ARCHIVE_SNAPSHOT` of each run's trace
  (mean over occupied cells).
- Prints three highlighted comparisons: `adaptive` vs `frozen_uniform`,
  `adaptive` vs `frozen_fixed`, and `random` vs `adaptive` (the last surfacing
  the per-operator archive sigma side by side, with the sigma difference and a
  paired `best_rank` head-to-head). A text table is always printed; `--report-json`
  additionally dumps the full structured report.

Main interactions:
- Summary mode (default): discovers `traces/sigma_control_*.json` (or files
  passed via `--summaries`) written by `experiment`/`run_sigma_control.py`, and
  reads each run row's `mapelites_sigma_mode`, `target`, `run_index`, `solved`,
  `best_rank`, `archive_occupancy`, and `trace_path`. The seed is reconstructed
  as `metadata.random_seed + run_index`. The per-operator archive sigma is read
  from the trace at `trace_path`, falling back to the summary row's
  `final_archive_sigma_{s,m,ml,l}` columns when that trace is unavailable.
- Trace mode (`--traces`, used for verification or when only traces exist):
  builds run records directly from each trace's `RUN_CONFIG`
  (`mapelites_sigma_mode`, `target`, `random_seed`, `run_index`),
  `ARCHIVE_SNAPSHOT` (occupancy and per-operator sigma), and `SOLVED`/`FAILED`
  events.

Subtleties:
- Pairing requires a `(target, seed)` key in both arms; runs missing either field
  (older traces that predate `run_index`/`mapelites_sigma_mode` in `RUN_CONFIG`)
  are reported as unpaired and fall into the `--default-mode` arm label
  (default `unknown`).
- A comparison whose arm is absent from the input is reported as missing and
  skipped rather than fabricated, so the output stays honest on partial batches.
- This script reads traces and summaries only; consistent with the analysis-script
  invariant, it modifies no solver behavior, game backends, or trace schemas.

### `scripts/compare_embedding_llm_closeness.py`

Per-target diagnostic that contrasts the local game's notion of "close" (the
embedding) with the LLM's, to expose solver blind spots.

Main function:
- For each target word it lists the top-N closest words two ways and prints them
  side by side: the embedding's nearest neighbors and the LLM's ordered list.
- Reports, per target: set overlap and overlap rate, exact-position matches and
  rate, Spearman rank correlation over shared words, the embedding-side BLIND
  SPOTS (embedding-close words the LLM never proposed), and an optional
  LLM-only-far list (LLM words the embedding ranks beyond `--far-rank` or places
  out of vocabulary). A batch over multiple targets prints a per-target summary
  row plus mean overlap / match rates; `--report-json` dumps the full report.

Main interactions:
- Embedding side: `EmbeddingModel.nearest_neighbors(target, n)` for the top-N
  neighbors (cosine, target excluded) - the same ranking `LocalGame` derives, so
  embedding-rank `r` is the r-th closest word the local game would rank.
- Actual embedding rank of an arbitrary LLM word: `LocalGame(model, target).rankings`
  (target = rank 1; missing words are out-of-vocab `n/a`), used only for the
  LLM-only-far check.
- LLM side: a task-specific ordered-list JSON prompt submitted through the public
  `LLMClient.complete_json_prompt()` (the same path `scripts/calibrate_anchors.py`
  uses, not `place_word`). Provider/model resolve from `config` with CLI overrides.

Subtleties:
- LLM words are normalized to single lowercase alphabetic words to match the
  game's guess constraints (phrases, hyphens, punctuation, and the target itself
  are dropped, duplicates removed); the report notes when fewer than N survive.
- LLM lists are cached per `(model, target, N)` (model in the cache filename,
  `target::nN` as the key) under `MAPELITES_PLACEMENT_CACHE_DIR`, so re-runs are
  free; `--no-llm` uses the cache only.
- The embedding is the LOCAL game's (MiniLM by default), not the real Contexto
  embedding, so its findings explain local-game behavior only.
- Analysis only: it reads `EmbeddingModel`, `LocalGame`, and `LLMClient` and
  modifies no solver, game, or config code.

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

### `contexto_solver.plot_map_elites`

Standalone visualization script for MAP-Elites (`method=ea_llm_map_elites`)
archive traces.

Main function:
- Reads an existing MAP-Elites trace JSON and renders seven static PNG figures:
  cell-occupancy heatmap, archive growth, cell hit-count heatmap, continuous
  placement scatter, per-component sigma heatmap (final), per-component sigma
  snapshots over time, and the winning-lineage sigma trajectory.
- Optional `--combined` writes an additional `map_elites_summary.png` montage of
  the generated figures.
- `--plots` selects a subset; `--snapshot-gens` controls the sigma snapshot
  timepoints (final is always appended).

Main interactions:
- Consumes generated traces only; it does not participate in solving or batch
  execution and needs no embedding model (MAP-Elites coordinates are already in
  the trace).
- Reads `AXIS_DEFINITION` for anchor scales and grid resolution, `PLACEMENT`
  for continuous coordinates, `ARCHIVE_PLACE/REPLACE/REJECT` for occupancy and
  hit counts, `ARCHIVE_SNAPSHOT` for archive state at a generation, and
  `INIT`/`OPERATOR_SAMPLED`/`CROSSOVER` for the lineage walk.

Subtleties:
- State-at-time plots read `ARCHIVE_SNAPSHOT` events directly (snapshots carry
  complete per-cell incumbent records), so reconstruction is a lookup, not an
  event replay. A PLACE/REPLACE replay is used only as a fallback before the
  first snapshot.
- Plot 4 recovers each placement's rank by pairing a `PLACEMENT` event with the
  immediately following `ARCHIVE_*` event, relying on the back-to-back emission
  order in the solver's `_place_and_compete`.
- The lineage walk is re-implemented here (not imported from the self-adaptive
  inspection script) so that script stays unmodified; it mirrors the same
  crossover parent resolution by category name and sigma.
- Output goes to a per-run subdirectory `figures/<run_label>/` (derived from the
  trace filename), which diverges from `plot_trajectory`'s flat `figures/`
  layout because a single MAP-Elites run produces seven-plus PNGs that benefit
  from grouping.
- On a non-MAP-Elites trace (no `AXIS_DEFINITION`), the script prints a clear
  message and exits without error rather than crashing.
- Plotting dependencies stay local to the module (Matplotlib is imported
  lazily); solver code never imports it.

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

### `scripts/measure_self_adaptive_selection_coupling.py`

Operator -> selection-survival and operator -> fitness coupling for plain
self-adaptive (`method=ea_llm_self_adaptive`) traces. It is the self-adaptive
counterpart to `scripts/measure_sigma_fitness_coupling.py`: that script links a
fired operator to a MAP-Elites archive (cell) win, whereas this one links it to
whether the mutation child survived selection. The fitness (delta) half is
identical (`delta = log(parent_rank) - log(child_rank)` on the log-rank scale).

Main function:
- Pools self-adaptive traces for the requested targets (default
  `herbaceous,notorious,superficial`) and, per fired mutation operator
  (`OPERATOR_SAMPLED.sampled_op`; crossover excluded), reports a per-operator
  survival rate and a per-operator delta-fitness distribution. `--by-word` adds
  a per-target split; `--report-json` dumps the full report.

Main interactions:
- Reads `OPERATOR_SAMPLED` (operator, `child_id`, `parent_id`,
  `child_hypothesis_name`), serialized hypothesis records from `INIT`/`MUTATE`/
  `CROSSOVER` (for `hypothesis_id` -> `best_rank`), and `SELECT`
  (`kept`/`discarded`/`elite`, by category name). Child rank is read by
  `child_id` (ID-reliable); parent rank is the parent's latest real rank at or
  before the child's event order.

Subtleties:
- "survived" means the child's `child_hypothesis_name` is in the NEXT
  generation's `SELECT.kept` (or is its `elite`). Within a generation the order
  is `... SELECT -> OPERATOR_SAMPLED -> MUTATE ...`, so a child is judged by the
  following generation's `SELECT`. This is the logged top-`max_active_hypotheses`+
  elite step; the separate `mu` cap (`SELF_ADAPTIVE_MU`) via
  `_cap_active_hypotheses` is unlogged and is NOT what this measures.
- Sentinel-rank children (`best_rank` 1e9 / None / <=0, i.e. no valid guess) are
  excluded from delta and counted as culled.
- A non-sentinel child is unresolvable only when it is a last-generation child
  (no next `SELECT`) or its name is reused by another mutation child in the same
  trace (ambiguous). Unresolvable children are dropped from the survival rate,
  and the per-operator unresolvable fraction is reported as a bias diagnostic.
  Absence from the next `SELECT` (removal by deduplication) is counted as culled,
  since `SELECT.kept`+`discarded` cover all live hypotheses.
- Analysis only: reads traces, runs no solver/new games, and changes no solver,
  game, or trace-schema code.

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
  `EmbeddingModel`, `LocalGame`, `methods/embedding.py`, README, and docs. The
  `EmbeddingModel.nearest_neighbors` / `LocalGame.rankings` contract is also
  relied on by `scripts/compare_embedding_llm_closeness.py`.
- LLM prompts or JSON schemas: update `LLMClient`, LLM methods'
  parsing/cleaning, trace expectations, and experiment notes. Scripts that own a
  prompt submitted through the public `complete_json_prompt()` path
  (`scripts/calibrate_anchors.py`, `scripts/compare_embedding_llm_closeness.py`)
  depend on that path staying stable.
- LLM provider routing or defaults: update `config`, `main`, `experiment`,
  `LLMClient`, `RUN_CONFIG` metadata, experiment summary fields, and smoke
  tests for both selected provider behavior and provider-specific errors.
- Selection, mutation, local search, or deduplication: update
  `methods/ea_core.py`, verify traces, and consider effects on
  convergence/diversity. The `SELECT` event shape (`kept`/`discarded`/`elite`
  by category name) and the within-generation event order are relied on by
  `scripts/measure_self_adaptive_selection_coupling.py` to recover survival.
- Self-adaptive operators or sigma behavior: update `operators.py`,
  `hypothesis.py`, `methods/ea_llm_self_adaptive.py`, prompt-leakage tests, and
  trace inspection/analysis scripts (`scripts/inspect_self_adaptive_trace.py`,
  `scripts/measure_self_adaptive_selection_coupling.py`).
- Experiment summary fields: update `experiment`, CSV fieldnames, downstream
  docs, and any scripts that read summaries, including
  `scripts/compare_sigma_control_arms.py` (which reads `mapelites_sigma_mode`,
  `target`, `run_index`, `solved`, `best_rank`, `archive_occupancy`,
  `trace_path`, and the `final_archive_sigma_*` columns).
- Trace event names/details: update solvers, `Logger` consumers, documentation,
  and any manual analysis assumptions.
- Analysis visualizations: update `contexto_solver.plot_trajectory`,
  `contexto_solver.plot_map_elites`, `requirements.txt`, and docs when trace
  interpretation, projection methods, or generated figure conventions change.
- MAP-Elites trace events (`AXIS_DEFINITION`, `PLACEMENT`, `ARCHIVE_*`): update
  `methods/ea_llm_map_elites.py`, `Logger`, and `contexto_solver.plot_map_elites`
  together, since the visualizer reconstructs archive state from those events.
  `ARCHIVE_SNAPSHOT` is additionally consumed by
  `scripts/compare_sigma_control_arms.py` (occupancy and per-operator sigma) and
  `scripts/measure_sigma_fitness_coupling.py`.

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
