# Experiment Log

This document collects the project's completed solver runs in one place. It is
intended as a readable companion to the raw JSON traces in `traces/`, not a
replacement for those traces.

## How To Read This Log

- `Solved` means the solver reached rank `1` in the local game, or the
  equivalent solved rank reported through the game interface.
- `Best rank` is the closest word found when a run did not solve the game.
- Local-game experiments use `data/glove.6B.300d.txt` unless otherwise noted.
- LLM runs are stochastic. Single runs are useful for debugging behavior, but
  paper-level claims should use the batch experiment runner over multiple
  targets and repeated runs.

## Completed Runs

### 2026-05-04, 12:12 — LLM Local Game, `cat`

- Command: `python main.py --game local --target cat --solver llm`
- Trace: `traces/llm_local_cat_20260504_121840.json`
- Result: solved.
- Final answer: `cat`.
- Total guesses: 468.
- Generations: 13.
- Important path: the solver found `dog` at rank 2, then local search around
  `dog` guessed `cat` at rank 1.
- Interpretation: local search works well once the LLM reaches a nearby word,
  but the broader evolutionary search spent many guesses in less useful food,
  cooking, and measurement hypotheses before reaching the animal neighborhood.

### 2026-05-04, 12:49 — LLM Real API Game, Game `1314`

- Command: `python main.py --game api --game-number 1314 --solver llm`
- Trace: `traces/llm_api_1314_20260504_124906.json`
- Result: not solved.
- Best word: `rug`.
- Best rank: 8.
- Total guesses: 290.
- Generations: 15.
- Important path: the solver converged on household/floor-covering language,
  with `rug` as a strong clue, but did not reach the answer before the
  generation limit.
- Interpretation: strong near-miss. This run motivated better control over
  hypothesis growth and better local exploitation.

### 2026-05-04, 13:37 — Embedding Smoke Test, Aligned, `cat`

- Command: batch experiment smoke run, aligned embedding mode.
- Summary: `traces/experiment_smoke.json`
- CSV: `traces/experiment_smoke.csv`
- Run trace: `traces/embedding_aligned_cat_run1_20260504_133724.json`
- Result: not solved.
- Best word: `rind`.
- Best rank: 24523.
- Total guesses: 4.
- Generations: 0.
- Configuration: solver `embedding`, mode `aligned`, target `cat`,
  `max_generations=0`, `random_seed=123`.
- Interpretation: this was a pipeline smoke test rather than a real benchmark.
  It verified that the batch experiment runner can produce JSON and CSV output.

### 2026-05-04, 14:00 — LLM Local Game, `cat`, After Stage 2b Fixes

- Command: `python main.py --game local --target cat --solver llm`
- Trace: `traces/llm_local_cat_20260504_140013.json`
- Result: solved.
- Final answer: `cat`.
- Total guesses: 2.
- Generations: 0.
- Important path: the initial `animals` category tried `dog` at rank 2 and
  then `cat` at rank 1 immediately.
- Interpretation: confirms immediate stopping works, but this is not a useful
  convergence benchmark because the target appeared in the starter words.

### 2026-05-04, 14:01 — LLM Local Game, `house`, After Stage 2b Fixes

- Command: `python main.py --game local --target house --solver llm`
- Trace: `traces/llm_local_house_20260504_140109.json`
- Result: solved.
- Final answer: `house`.
- Total guesses: 107.
- Generations: 4.
- Important path: mutation discovered a legislative-body interpretation;
  `senate` reached rank 3 and then `house` reached rank 1.
- Interpretation: a better stress test than `cat`. It shows pivot-aware
  mutation can move from a broad clue to a different sense of the target.

### 2026-05-04, 14:15 — LLM Real API Game, Game `1314`

- Command: `python main.py --game api --game-number 1314 --solver llm`
- Trace: `traces/llm_api_1314_20260504_141552.json`
- Result: solved.
- Final answer: `ivory`.
- Total guesses: 165.
- Generations: 7.
- Important path: `bead` reached rank 19, then mutation explored smooth white
  objects and guessed `ivory`.
- Interpretation: confirms the post-fix LLM evolutionary pipeline can solve a
  real Contexto game and stop immediately after the correct answer is found.

### 2026-05-04, 14:38 — LLM Local Game, `notorious`, 15 Generations

- Command: `python main.py --game local --target notorious --solver llm`
- Trace: `traces/llm_local_notorious_20260504_143848.json`
- Result: not solved.
- Best word: `crime`.
- Best rank: 19.
- Total guesses: 291.
- Generations: 15.
- Important path: the solver converged on law enforcement and crime language,
  but local search around `crime` proposed narrow legal terms such as
  `offense`, `felony`, `misdemeanor`, `theft`, and `burglary`.
- Interpretation: this run showed that the old default generation budget could
  stop the search while it was still near the target neighborhood.

### 2026-05-04, 15:22 — LLM Local Game, `notorious`, 20 Generations

- Command: `python main.py --game local --target notorious --solver llm`
- Trace: `traces/llm_local_notorious_20260504_152234.json`
- Result: not solved.
- Best word: `gang`.
- Best rank: 4.
- Total guesses: 281.
- Generations: 20.
- Important path: increasing `MAX_GENERATIONS` from 15 to 20 allowed the run
  to improve from `crime` rank 19 to `gang` rank 4.
- Observed issue: later generations wasted proposals on duplicates. Debug logs
  showed examples such as generation 13 with `rawCount=15`, `acceptedCount=0`,
  and `rejectedDuplicate=15`.
- Observed issue: local search around `gang` repeatedly suggested close group
  nouns such as `crew`, `group`, `clique`, `mob`, and `pack`, but did not
  explore enough descriptor-style words.
- Follow-up change: the local-search prompt was broadened to ask for related
  descriptors, collocations, associated people/groups, causes/effects, and
  nearby-context words. Candidate generation also now passes the global guess
  history to the LLM so it can avoid words already tried by other hypotheses.
- Verification status: completed in the follow-up runs below.

### 2026-05-04, 15:38 — LLM Local Game, `notorious`, Broader Local Search

- Command: `python main.py --game local --target notorious --solver llm`
- Trace: `traces/llm_local_notorious_20260504_153816.json`
- Result: not solved.
- Best word: `gang`.
- Best rank: 4.
- Total guesses: 380.
- Generations: 20.
- Important path: the broader prompt and global avoid list helped the solver
  find better intermediate clues, including `smuggler` rank 42 and `gangster`
  rank 9, but the run still converged to `gang` rank 4.
- Interpretation: local search improved exploration quality but still tended
  to circle around criminal-group nouns once it reached the `gang` area.

### 2026-05-04, 15:54 — LLM Real API Game, Game `1323`

- Command: `python -m contexto_solver.main --game-number 1323 --max-generations 15`
- Trace: `traces/llm_api_1323_20260504_155456.json`
- Result: solved.
- Final answer: `sponges`.
- Total guesses: 254.
- Generations: 15.
- Important path: the solver reached `coral` at rank 10, explored reef and
  ocean-related hypotheses, and then guessed `sponges` from the `reef animals`
  hypothesis.
- Interpretation: the online API solver can still solve after the recent
  control changes, but this example exceeded the desired sub-200 guess range.

### 2026-05-04, 16:26 — LLM Local Game, `notorious`, Earlier Local Search

- Command: `python main.py --game local --target notorious --solver llm`
- Trace: `traces/llm_local_notorious_20260504_162627.json`
- Result: not solved.
- Best word: `gang`.
- Best rank: 4.
- Total guesses: 387.
- Generations: 20.
- Important path: after raising the local-search threshold to 100 and adding a
  retry for duplicate-only local-search responses, the solver still converged
  to organized-crime language. The final active area included `gang`, `kingpin`,
  `gangster`, and `mafia`, but not the target `notorious`.
- Interpretation: earlier local search did not solve this target. The remaining
  weakness is not just when local search starts, but how the solver pivots from
  a close noun (`gang`) to associated descriptors or reputational adjectives.

### 2026-05-05, 12:26 — LLM Local Game, `herbaceous`, Fast Solve

- Command: `python main.py --game local --target herbaceous --solver llm`
- Trace: `traces/llm_local_herbaceous_20260505_122620.json`
- Result: solved.
- Final answer: `herbaceous`.
- Total guesses: 101.
- Generations: 4.
- Important path: the solver reached plant-related hypotheses quickly and
  guessed `herbaceous` from `plant types`.
- Interpretation: this is the best of the three repeated `herbaceous` runs and
  shows the solver can solve this local GloVe target efficiently when the LLM
  takes the right plant-category path.

### 2026-05-05, 13:02 — LLM Local Game, `herbaceous`, Rank-3 Stall

- Command: `python main.py --game local --target herbaceous --solver llm`
- Trace: `traces/llm_local_herbaceous_20260505_130240.json`
- Result: not solved.
- Best word: `shrub`.
- Best rank: 3.
- Total guesses: 311.
- Generations: 20.
- Important path: the runtime summaries showed `shrub` at rank 3 by generation
  4, then the run stayed at `shrub` through generation 20.
- Interpretation: this is a strong example of local convergence failure. The
  solver found an extremely close clue but did not make the final jump to the
  descriptor `herbaceous`.

### 2026-05-05, 13:05 — LLM Local Game, `herbaceous`, Later Solve

- Command: `python main.py --game local --target herbaceous --solver llm`
- Trace: `traces/llm_local_herbaceous_20260505_130555.json`
- Result: solved.
- Final answer: `herbaceous`.
- Total guesses: 215.
- Generations: 10.
- Important path: the run reached `vine` at rank 83, then `foliage` at rank
  47, then `shrub` at rank 3, and eventually guessed `herbaceous` from the
  `small plants` hypothesis.
- Interpretation: this run sits between the fast solve and the stall. It
  confirms that reaching `shrub` rank 3 can still lead to the target, but only
  if the solver explores the right descriptor relation afterward.

### 2026-05-06, 15:00 — LLM Local Game, `superficial`, Obvious/Subtle Stall

- Command: `python main.py --game local --target superficial --solver llm`
- Trace: `traces/llm_local_superficial_20260506_150023.json`
- Result: not solved.
- Best word: `obvious`.
- Best rank: 8.
- Total guesses: 383.
- Generations: 20.
- Important path: the run moved into the `subtle`/`obvious` neighborhood and
  held strong ranks there, but did not make the final jump to `superficial`.
- Interpretation: the solver found antonym/contrast vocabulary that is close
  in the local embedding space, but the feedback did not reliably guide it
  toward the target adjective.

### 2026-05-06, 15:10 — LLM Local Game, `superficial`, Anatomy Drift

- Command: `python main.py --game local --target superficial --solver llm`
- Trace: `traces/llm_local_superficial_20260506_151008.json`
- Result: not solved.
- Best word: `visceral`.
- Best rank: 41.
- Total guesses: 316.
- Generations: 20.
- Important path: the run went through anatomy/body terms: `head`, `thigh`,
  `pelvis`, `abdomen`, and then `visceral`.
- Interpretation: this is a misleading-rank case. The local GloVe backend gave
  useful-looking feedback for an anatomy/internal-organ direction, but that
  trajectory was semantically wrong for the target `superficial`.

### 2026-05-06, 15:15 — LLM Local Game, `superficial`, Subtle Stall

- Command: `python main.py --game local --target superficial --solver llm`
- Trace: `traces/llm_local_superficial_20260506_151506.json`
- Result: not solved.
- Best word: `subtle`.
- Best rank: 13.
- Total guesses: 348.
- Generations: 20.
- Important path: the run moved from consequence/implication language into
  `subtle`, then stayed at `subtle` from generation 13 through generation 20.
- Interpretation: like the 15:00 run, this found a plausible abstract adjective
  neighborhood but failed to pivot from related contrast language to the exact
  surface/depth relation.

### 2026-05-07, 13:33 — LLM Local Game, `superficial`, Ollama Solve

- Command: `python main.py --game local --target superficial --solver llm --provider ollama --ollama-model qwen3:14b`
- Trace: `traces/llm_local_superficial_20260507_133325.json`
- Result: solved.
- Final answer: `superficial`.
- Total guesses: 388.
- Generations: 15.
- Configuration: provider `ollama`, model `qwen3:14b`, `max_generations=50`.
- Important path: the run moved from broad emotion/body clues through `skin`
  rank 40 and `epidermis` rank 21, then reached `vein` at rank 3. Local search
  around `vein` proposed `superficial` and solved the target.
- Interpretation: the local Ollama path can solve a previously stalled local
  target when given a larger generation budget. The successful bridge came
  through an anatomy/surface-depth relation rather than the earlier
  `subtle`/`obvious` adjective neighborhood.

## Cross-Run Observations

- Local search is valuable when the best clue is extremely close. The `cat`
  run solved immediately from `dog` rank 2, and the real API run reached
  `ivory` after moving through nearby smooth/white-object clues.
- Pivot-aware mutation matters. The `house` run succeeded because the solver
  moved from ordinary categories into the legislative sense of `house`.
- Generation budget matters for difficult targets. `notorious` improved from
  rank 19 at 15 generations to rank 4 at 20 generations.
- Duplicate proposals become a major efficiency problem late in a run. This is
  especially visible after the solver converges on a strong neighborhood and
  keeps asking the LLM for more words in that same area.
- The online API appears easier for the LLM evolutionary solver than the local
  GloVe game in several current runs, even though the solver does not directly
  inspect either backend's embedding model.
- Repeated same-target local runs can vary sharply. Three `herbaceous` runs
  with the same local game and LLM setup produced a fast solve, a rank-3 stall,
  and a later solve.
- Single-run outcomes are highly variable. The same solver can solve trivial
  cases during initialization, solve semantic pivots in a few generations, or
  remain stuck near a strong clue.

