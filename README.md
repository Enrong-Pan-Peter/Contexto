# Contexto Evolutionary Solver

This project studies automated solvers for
[Contexto](https://contexto.me/), a word guessing game where each guess receives
a semantic rank. Lower ranks are closer to the hidden answer.

The current research codebase compares several solver methods, embedding-neighbor
baselines, local embedding-backed games, and real Contexto API games while
preserving readable JSON traces of the search process.

## Current Status

Implemented:

- Local offline Contexto-style game using static embedding matrices.
- Manual terminal play for the local game.
- Method-based solver architecture with pure LLM, EA+LLM, EA+LLM+pivot, and
  embedding-neighbor methods.
- Embedding-neighbor baseline solver.
- Batch local experiment runner with resumable JSON/CSV summaries.
- Pivot-matrix analysis tooling for paired pivot-on/off experiments.
- Real Contexto API wrapper with rank normalization.
- OpenAI, Anthropic, and Ollama LLM provider support.
- Shared game interface used by both local and API games.
- JSON trace logging for solver runs.
- Parallel LLM generation with configurable worker count.
- Separate local game and solver embedding paths for aligned/non-aligned tests,
  including GloVe, MiniLM, and MPNet cache paths.
- Mitigations for duplicate hypotheses, invalid guesses, singular/plural
  redundancy, and local semantic stagnation.

Not yet complete:

- Full-size MiniLM and MPNet cache generation/benchmarking.
- Batch experiment runner for real API games.
- Continuous pivot strategy.

## Architecture

```text
contexto_solver/
  embeddings.py          Load static embedding matrices and query neighbors
  build_embedding_cache.py
                         Build MiniLM/MPNet word-vector caches
  local_game.py          Offline Contexto-style game using embeddings
  play.py                Manual terminal interface for the local game
  game_api.py            Real Contexto API wrapper
  methods/
    llm_only.py          Pure LLM baseline, one history-conditioned guess at a time
    ea_llm.py            EA+LLM method without stall pivots
    ea_llm_pivot.py      EA+LLM method with stall-pivot operators
    embedding.py         Embedding-neighbor baseline
    ea_core.py           Shared EA+LLM core
  llm_client.py          OpenAI/Anthropic/Ollama API wrapper
  hypothesis.py          Hypothesis/category model for LLM solver
  logger.py              JSON trace logger
  config.py              Environment and default settings
  main.py                Main CLI entry point

main.py                  Root wrapper for contexto_solver.main
play.py                  Root wrapper for contexto_solver.play
docs/                    Architecture, findings, timeline, and experiment docs
traces/                  Generated JSON traces
data/                    Local embedding files, not committed
```

All automatic methods talk to a game object through this shared interface:

```python
guess(word) -> int
total_guesses() -> int
best_so_far() -> tuple[str | None, int | None]
is_solved() -> bool
```

For the local game, rank `1` is the target word. For the real Contexto API, the
API returns `0` for the answer, and `game_api.py` normalizes that to rank `1` so
the solvers can use the same interface.

## Setup

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Download GloVe vectors.

Use the Stanford pretrained vectors:

```text
Wikipedia 2014 + Gigaword 5: glove.6B.zip
```

Download page:

```text
https://nlp.stanford.edu/projects/glove/
```

After unzipping, place this file at:

```text
data/glove.6B.300d.txt
```

3. Optionally build transformer embedding caches.

Sentence-transformer models do not ship as fixed word-vector lists like GloVe.
The project precomputes a chosen vocabulary into the same runtime interface used
by the local game and embedding solver.

Build a MiniLM cache from the GloVe vocabulary:

```powershell
python -m contexto_solver.build_embedding_cache --model sentence-transformers/all-MiniLM-L6-v2 --vocab-source data/glove.6B.300d.txt --output data/embeddings/all-MiniLM-L6-v2.npz --limit 200000
```

Build the heavier MPNet cache:

```powershell
python -m contexto_solver.build_embedding_cache --model sentence-transformers/all-mpnet-base-v2 --vocab-source data/glove.6B.300d.txt --output data/embeddings/all-mpnet-base-v2.npz --limit 200000
```

The `--limit` is optional. It is useful for first validation because full
400,000-word transformer caches can take longer to build and load.

4. Create a local `.env` file.

Use `.env.example` as a starting point:

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.4-mini
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3:14b
OLLAMA_REQUEST_TIMEOUT_SECONDS=900
LLM_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LLM_WORKERS=4
GLOVE_PATH=data/glove.6B.300d.txt
EMBEDDING_CACHE_DIR=data/embeddings
MINILM_EMBEDDING_PATH=data/embeddings/all-MiniLM-L6-v2.npz
MPNET_EMBEDDING_PATH=data/embeddings/all-mpnet-base-v2.npz
# GAME_EMBEDDING_PATH=data/embeddings/all-MiniLM-L6-v2.npz
# SOLVER_EMBEDDING_PATH=data/embeddings/all-MiniLM-L6-v2.npz
TRACE_DIR=traces
MAX_GENERATIONS=50
INITIAL_CATEGORIES=6
MAX_ACTIVE_HYPOTHESES=5
LOCAL_SEARCH_RANK_THRESHOLD=100
SELF_ADAPTIVE_MU=15
SELF_ADAPTIVE_INITIAL_CATEGORIES=15
EA_LLM_PIVOT_STALL_NO_IMPROVEMENT_GENERATIONS=3
EA_LLM_PIVOT_STALL_CLOSE_RANK_THRESHOLD=30
EA_LLM_PIVOT_STALL_CLOSE_GENERATIONS_LIMIT=5
EA_LLM_PIVOT_MAX_ATTEMPTS_PER_RUN=5
EA_LLM_PIVOT_CANDIDATE_WORDS_PER_OPERATOR=10
EA_LLM_PIVOT_RESOLUTION_WINDOW=2
EMBEDDING_SEED_COUNT=12
EMBEDDING_ACTIVE_COUNT=5
EMBEDDING_NEIGHBORS_PER_WORD=10
RANDOM_SEED=
```

For OpenAI, set either `LLM_API_KEY` or `OPENAI_API_KEY`. Ollama runs use the
local Ollama server and do not require a cloud API key.

## Manual Local Game

Play a local Contexto-style game manually:

```powershell
python play.py cat
```

Useful commands while playing:

- Type a word to guess.
- Type `hint` to see your best guessed words so far.
- Type `quit` to reveal the target.

## Automatic Methods

### Available Methods

- `llm_only`: pure LLM baseline. It asks the LLM for one new guess from the
  running `(word, rank)` history. No hypotheses, mutation, crossover, or local
  search.
- `ea_llm`: evolutionary LLM method with hypotheses, selection, mutation,
  crossover, deduplication, and LLM local search. No stall-pivot operators.
- `ea_llm_pivot`: `ea_llm` plus stall detection and pivot operators for
  morphology, register shifts, and adjacent-category jumps.
- `ea_llm_self_adaptive`: EA+LLM method whose hypotheses carry mutation
  operator probabilities. It uses `SELF_ADAPTIVE_MU` and
  `SELF_ADAPTIVE_INITIAL_CATEGORIES` instead of the regular EA population
  defaults.
- `ea_llm_map_elites`: MAP-Elites variant of `ea_llm_self_adaptive`. It replaces
  top-mu selection with a `5x5` behavior archive over two LLM-placed axes
  (concreteness and specificity), keeping the inherited self-adaptive sigma
  machinery. Tuned with `MAPELITES_*` config values.
- `embedding`: embedding nearest-neighbor baseline.

The `solver` field in traces remains a broad compatibility label (`llm` or
`embedding`). Use the `method` field to distinguish `llm_only`, `ea_llm`,
`ea_llm_pivot`, `ea_llm_self_adaptive`, and `ea_llm_map_elites`.

### EA+LLM Against Local Game

```powershell
python main.py --game local --target cat --method ea_llm_pivot
```

This uses the configured local embedding backend, but the guesses (categories
and candidate words) are generated by the LLM. If a MiniLM cache exists and
`GAME_EMBEDDING_PATH` is unset, MiniLM is used as the default local backend;
otherwise the project falls back to GloVe.

You can increase parallel LLM calls:

```powershell
python main.py --game local --target cat --method ea_llm_pivot --llm-workers 8
```

The current EA+LLM methods include several mitigations added after early slow,
wandering, and stalled runs:

- Active hypotheses are capped at `MAX_ACTIVE_HYPOTHESES`.
- Near-duplicate hypotheses are merged.
- Mutation prompts ask for divergent interpretations of strong clues, not only
  narrower sub-categories.
- Candidate prompts receive global guess history to reduce duplicate proposals.
- Local search starts when the best rank is below
  `LOCAL_SEARCH_RANK_THRESHOLD`.
- `ea_llm_pivot` can trigger pivot operators for morphology, register shifts,
  and adjacent-category jumps.
- Singular/plural variants of already tried words are filtered as redundant.
- Invalid or unrecognized guesses are remembered and avoided.

### Pure LLM Against Local Game

```powershell
python main.py --game local --target cat --method llm_only
```

This is the simplest LLM baseline. It is useful as a comparison point because it
uses rank history only and does not run the evolutionary pipeline.

### Embedding Method Against Local Game

```powershell
python main.py --game local --target cat --method embedding
```

This uses the same configured embedding model for the local game and for
generating nearest-neighbor guesses. This is the aligned embedding baseline.

Available embedding choices are:

- `GLOVE_PATH`: legacy static GloVe baseline.
- `MINILM_EMBEDDING_PATH`: lightweight modern sentence-transformer cache.
- `MPNET_EMBEDDING_PATH`: heavier sentence-transformer cache.

The separate `GAME_EMBEDDING_PATH` and `SOLVER_EMBEDDING_PATH` settings support
aligned and non-aligned comparisons.

### LLM Method Against Real Contexto API

```powershell
python main.py --game api --game-number 1314 --method ea_llm_pivot
```

The real API uses `API_RATE_LIMIT` in `config.py`. Local games do not use this
delay.

### Embedding Method Against Real Contexto API

```powershell
python main.py --game api --game-number 1314 --method embedding
```

This tests whether local GloVe neighbors can still guide search when the real
game likely uses a different embedding model.

## CLI Options

```text
--game {local,api}              Game backend
--method {llm_only,ea_llm,ea_llm_pivot,ea_llm_self_adaptive,ea_llm_map_elites,embedding}
                                  Solver method
--target TARGET                 Target word for local game
--game-number GAME_NUMBER       Real Contexto game number
--max-generations N             Generation budget
--provider {openai,anthropic,ollama}
                                  LLM provider
--model MODEL                   LLM model name
--ollama-model MODEL            Ollama model name
--api-key API_KEY               LLM API key override
--glove-path PATH               GloVe embedding file path
--game-embedding-path PATH      Embedding file used by LocalGame
--solver-embedding-path PATH    Embedding file used by embedding method
--llm-workers N                 Parallel LLM generation calls
--seed-count N                  Random seed words for embedding solver
--active-count N                Active words retained by embedding solver
--neighbors-per-word N          Nearest neighbors queried per active word
--random-seed N                 Reproducible embedding solver seed
```

## Batch Experiments

Run local aligned embedding experiments:

```powershell
python -m contexto_solver.experiment --targets cat,dog,ivory --mode aligned --method embedding --random-seed 123
```

Run local non-aligned embedding experiments by using different embedding files
for the local game and solver:

```powershell
python -m contexto_solver.experiment --targets cat,dog,ivory --mode non_aligned --method embedding --game-embedding-path data/embeddings/all-MiniLM-L6-v2.npz --solver-embedding-path data/glove.6B.300d.txt
```

This compares a MiniLM local-game backend against a GloVe embedding solver. You
can also set `--solver-embedding-path data/embeddings/all-mpnet-base-v2.npz`
after building the MPNet cache.

The experiment runner writes:

- One trace JSON file per run in `traces/`.
- One summary JSON file.
- One summary CSV file.

Current batch experiments focus on local games. Real API batch experiments are
left for future work because they need stricter rate-limit and game-selection
controls.

Pivot-on/off local LLM matrices can be analyzed with:

```powershell
python -m contexto_solver.analyze_pivot_matrix --off traces/pivot_matrix_off.json --on traces/pivot_matrix_on.json --output-prefix traces/pivot_matrix
```

After the method refactor, use `--method ea_llm` for pivot-off and
`--method ea_llm_pivot` for pivot-on. The experiment summaries still include
`enable_pivot` compatibility metadata so the existing analysis script can pair
and analyze them.

For run evidence and research-facing interpretation, see:

- `docs/experiment_log.md`
- `docs/findings.md`

## Traces

Each solver run saves a readable JSON trace in `traces/`.

Trace events include:

- `INIT`: initial categories or seed words.
- `GUESS`: a submitted word and returned rank.
- `SELECT`: selected active hypotheses or words.
- `MUTATE`: specialized child hypotheses from EA+LLM methods.
- `CROSSOVER`: combined hypotheses from EA+LLM methods.
- `LOCAL_SEARCH`: focused guesses near a strong word.
- `PIVOT_TRIGGERED`: pivot operator results for `ea_llm_pivot`.
- `PIVOT_RESOLUTION`: whether a pivot was followed by rank improvement.
- `AXIS_DEFINITION`, `PLACEMENT`, `ARCHIVE_PLACE`/`ARCHIVE_REPLACE`/`ARCHIVE_REJECT`,
  `ARCHIVE_SNAPSHOT`: MAP-Elites archive events for `ea_llm_map_elites`.
- `SOLVED`: answer found.
- `FAILED`: generation budget exhausted.

Example trace path:

```text
traces/llm_local_cat_20260504_121840.json
```

## Visualization And Inspection

Trajectory plots (rank, distance, 2D projection) from any trace:

```powershell
python -m contexto_solver.plot_trajectory --plot-type rank --trace traces/<run_label>.json
```

MAP-Elites archive visualizations (seven PNGs written to `figures/<run_label>/`):

```powershell
python -m contexto_solver.plot_map_elites --trace traces/<map_elites_run>.json --combined
```

Use `--plots occupancy,scatter` to render a subset and `--snapshot-gens 10,20,30,40`
to choose the sigma snapshot timepoints. Running it on a non-MAP-Elites trace
exits cleanly with a message. Self-adaptive sigma/operator inspection lives in
`scripts/inspect_self_adaptive_trace.py`.

## Validation Performed

The local embedding game was initially validated with GloVe:

```text
VOCAB 400000
SHAPE (400000, 300)
KING_NEIGHBORS [('queen', ...), ('prince', ...), ('monarch', ...)]
RANKS {'cat': 1, 'dog': 2, 'car': 3417, 'asdfgh': -1}
STAGE1_VALIDATION_OK
```

A local LLM run also solved target `cat`:

```text
python main.py --game local --target cat --method ea_llm_pivot
Status: SOLVED
Best word: cat
Best rank: 1
Total guesses: 468
Generations: 13
```

Recent documented runs show the current behavior more clearly:

- Real API game `1314` solved as `ivory` in 165 guesses over 7 generations.
- Real API game `1323` solved as `sponges` in 254 guesses over 15 generations.
- Local GloVe target `house` solved in 107 guesses over 4 generations.
- Local GloVe target `notorious` remained difficult: recent runs reached
  `gang` at rank 4 but did not solve within 20 generations.

## Documentation Map

- `docs/architecture.md`: source of truth for project structure and invariants.
- `docs/design_decisions.md`: algorithmic and experimental design rationale.
- `docs/experiment_log.md`: compact run and batch evidence register.
- `docs/findings.md`: paper-facing findings and evidence-quality notes.
- `docs/research_timeline.md`: chronological project timeline.

## Notes For Future Work

The local batch experiment runner can now compare:

- Pure LLM, EA+LLM, EA+LLM+pivot, and embedding methods on local games.
- Embedding method with aligned game/solver embeddings.
- Embedding method with non-aligned game/solver embeddings.
- GloVe, MiniLM, and MPNet embedding choices once the corresponding caches are
  available.

Future work should extend batch experiments to real API games, add broader
aggregate reports across local and API benchmark sets, and benchmark the new
MiniLM/MPNet caches against GloVe after full cache generation.
