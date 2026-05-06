# Findings and Decision Log

This document tracks experiments, results, and design decisions for the
Contexto evolutionary solver project. Entries are in reverse chronological
order (newest first).


## 2026-05-06 — `herbaceous` runs expose singular/plural stall

**Setup:** Local GloVe game (`glove.6B.300d`), LLM evolutionary solver on the
`stall` branch, target `herbaceous`.

**Runs:**
- `python main.py --game local --target herbaceous --solver llm`
  - Result: not solved after 274 guesses over 20 generations.
  - Trace: `traces/llm_local_herbaceous_20260506_141417.json`
  - Best word: `shrubs`, rank 2.
  - Path: `tree -> plant -> vegetation -> shrub -> shrubs`, then stalled at
    `shrubs` from generation 5 through generation 20.
- `python main.py --game local --target herbaceous --solver llm`
  - Result: solved in 219 guesses over 10 generations.
  - Trace: `traces/llm_local_herbaceous_20260506_142033.json`
  - Path: reached `shrub` at rank 3 by generation 4, then later guessed
    `herbaceous` from the `plant forms` hypothesis.
- `python main.py --game local --target herbaceous --solver llm`
  - Result: not solved after 345 guesses over 20 generations.
  - Trace: `traces/llm_local_herbaceous_20260506_142357.json`
  - Best word: `shrub`, rank 3.
  - Path: reached `shrub` at generation 4 and stayed there through generation
    20.

**Observations:**
- The first run is especially important because `shrubs` is only the plural of
  `shrub`. Contexto effectively treats singular/plural forms as the same guess,
  so spending search effort on `shrub -> shrubs` is not a meaningful semantic
  pivot.
- The run that solved did so by entering a descriptor/register path (`plant
  forms`) and eventually guessing `herbaceous`, not by staying inside simple
  noun variants.
- This suggests the stall pivot mechanism needs to avoid singular/plural
  variants and focus on relation changes such as descriptors, botanical
  terminology, growth habit, and lexical register shifts.

**Change motivated by this finding:**
- Updated LLM prompts to tell the model not to propose singular/plural variants
  of already tried words.
- Added lightweight singular/plural family filtering in the LLM solver so
  obvious variants like `shrub`/`shrubs`, `bush`/`bushes`, and
  `berry`/`berries` are treated as redundant during candidate acceptance.


## 2026-05-05 — Repeated local LLM runs for `herbaceous`

**Setup:** Local GloVe game (`glove.6B.300d`), LLM evolutionary solver with
the same configured LLM and same local backend. The target word was
`herbaceous` in all three runs.

**Runs:**
- `python main.py --game local --target herbaceous --solver llm`
  - Result: solved in 101 guesses over 4 generations.
  - Trace: `traces/llm_local_herbaceous_20260505_122620.json`
  - Final path: the solver guessed `herbaceous` from the `plant types`
    hypothesis.
- `python main.py --game local --target herbaceous --solver llm`
  - Result: not solved after 311 guesses over 20 generations.
  - Trace: `traces/llm_local_herbaceous_20260505_130240.json`
  - Best word: `shrub`, rank 3.
  - Notable behavior: by generation 4 the solver reached `shrub` at rank 3,
    then stayed at `shrub` through generation 20 without converging to
    `herbaceous`.
- `python main.py --game local --target herbaceous --solver llm`
  - Result: solved in 215 guesses over 10 generations.
  - Trace: `traces/llm_local_herbaceous_20260505_130555.json`
  - Final path: after reaching `shrub` at rank 3, the solver eventually
    explored `small plants` and guessed `herbaceous`.

**Observations:**
- These three runs used the same target, same local game, and same LLM setup,
  but produced very different outcomes: fast solve, rank-3 stall, and later
  solve.
- The second run is especially interesting because `shrub` at rank 3 is very
  close to the answer, but the search still failed to make the final semantic
  jump.
- This suggests that the remaining variance is not only about finding a good
  neighborhood. The solver also needs a more reliable way to pivot from a very
  close clue to the exact target relation or descriptor.

**Possible follow-up:**
- Compare the traces around the first appearance of `shrub` across all three
  runs to identify what made one run choose `herbaceous` quickly while another
  stayed stuck.
- Use this target as a small case study for stochastic LLM variance,
  exploration/exploitation balance, and local-search failure modes.


## 2026-05-04 — Performance after fixes (cap, dedup, pivot-aware mutation)

**Setup:** Tested both online Contexto API and local GloVe game with LLM
evolutionary solver (GPT-5.4-mini), Stage 2b features enabled.

**Changes applied since last run:**
- Capped active hypotheses at 5 per generation.
- Added deduplication of near-identical hypotheses.
- Changed mutation prompt to encourage divergent interpretations of
  high-scoring words instead of always specializing deeper.

**Results:**

Online Contexto + LLM + evolutionary:
- Before fixes: ~500 guesses, 15 generations at worst.
- After fixes: under 200 guesses, 11 generations at worst in the first
  post-fix verification set. A later API game `1323` solved in 254 guesses
  over 15 generations, so the practical worst case is still higher on some
  targets.

Local GloVe game + LLM + evolutionary:
- Before fixes: 600+ guesses, 20+ generations at worst.
- After fixes: ~300 guesses, 15 generations. Still worse than online.
- Later `notorious` local runs improved to rank 4 but did not solve within
  20 generations, with 380-387 guesses.

**Observations:**
- The fixes reduced guess count significantly for both setups.
- The local game consistently performs worse than the online game.
  This is unexpected because the LLM solver does not use any embedding
  model directly, so the backend should not matter. Possible explanation:
  GPT-5.4-mini's internal sense of word similarity aligns better with
  whatever embedding model the real Contexto uses than with GloVe.
  If true, this is an interesting finding about implicit manifold
  alignment between LLMs and specific embedding models.
- Variance across games is still high (60 to 500 guesses depending on
  the target word).
- Convergence is sometimes poor. In some runs the best rank is 60 in
  generation 0 but only improves to 40 by generation 10, meaning 10
  generations of search barely improved the result.


## 2026-05-04 — First LLM solver run (Stage 2b, target: cat)

**Setup:** Local GloVe game (glove.6B.300d), LLM solver with GPT-5.4-mini,
Stage 2b features enabled (crossover, local search, elitism).

**Result:** Solved in 468 guesses over 13 generations.

**Observations:**
- By generation 7, there were 18 active hypotheses due to unchecked
  mutation. Most guesses were wasted on redundant categories.
- Mutation created near-duplicate categories. "food" spawned "dimensions
  of food", "measurements of food", "measurements in cooking", "portion",
  "portions", "portion size" which all explore the same area.
- "bite" scored rank 67 but the solver only explored food-related meanings.
  It missed that "bite" also relates to animals (the target was "cat").
  The solver went deep into food/cooking for 7+ generations before
  reaching the animal neighborhood.
- "dog" was guessed at rank 2 in generation 13. Local search immediately
  found "cat" from "dog". Local search works well once it has a very
  nearby word, but the broader search was too slow getting there.
- 468 guesses across 13 generations = ~36 guesses per generation, far
  too many due to the bloated hypothesis population.

**Changes made after this run:**
1. Capped active hypotheses at 5 per generation.
2. Added hypothesis deduplication (merge near-identical categories).
3. Changed mutation prompt to encourage divergent exploration.


## 2026-05-04 — Post-fix verification runs for active cap, deduplication, and pivot mutation

**Setup:** Local GloVe game (`glove.6B.300d`), LLM solver, Stage 2b features
enabled, active hypothesis cap set to 5.

**Runs:**
- `python main.py --game local --target cat --solver llm`
  - Result: solved in 2 guesses over 0 generations.
  - Trace: `traces/llm_local_cat_20260504_140013.json`
- `python main.py --game local --target house --solver llm`
  - Result: solved in 107 guesses over 4 generations.
  - Trace: `traces/llm_local_house_20260504_140109.json`

**Observations:**
- The `cat` run solved during initialization because the LLM included `cat`
  among the first starter words. This confirms the solver can stop immediately,
  but it is not a useful convergence benchmark.
- The `house` run is a better stress test. It solved under the target threshold
  of 200 guesses.
- `SELECT` events in the `house` trace kept at most 5 hypotheses.
- `DEDUPLICATE` events appeared in the `house` trace, confirming that similar
  hypotheses were merged.
- The mutation prompt produced more divergent directions. Examples included
  `command and authority`, `front part`, `government buildings`, `workplace or
  job sense`, and `animal or insect senses`, rather than only narrower
  sub-categories of the parent theme.

**Changes confirmed:**
1. Active hypothesis cap limits the number of categories proposing words in
   each generation.
2. Deduplication reduces redundant hypothesis accumulation.
3. Pivot-aware mutation encourages alternate interpretations of high-scoring
   words.

**Caution:** These are stochastic LLM runs. A single run should not be reported
as a stable performance estimate. The batch experiment runner should be used
for averages over multiple targets and repeated runs.


## 2026-05-04 — First LLM solver run (Stage 2b, target: cat)

**Setup:** Local GloVe game (`glove.6B.300d`), LLM solver with GPT-5.4-mini,
Stage 2b features enabled (crossover, local search, elitism).

**Result:** Solved in 468 guesses over 13 generations.

**Observations:**
- The solver started with 6 broad categories. By generation 7, there were
  18 active hypotheses due to unchecked mutation.
- Mutation created many near-duplicate categories. For example, `food`
  spawned `dimensions of food`, `measurements of food`, `measurements in
  cooking`, `portion`, `portions`, and `portion size`, which all explore
  essentially the same semantic area.
- The word `bite` scored rank 67 in generation ~5, which is a useful signal.
  However, mutation only explored food-related interpretations of `bite`
  (portions, cooking) and missed the animal interpretation (`things that
  bite`) until much later.
- `dog` was finally guessed in generation 13 at rank 2. Local search
  immediately found `cat` from `dog`. This confirms local search works well
  once it has a nearby word, but the broader search took too long to reach
  the right semantic neighborhood.
- Most of the 468 guesses were spent on food, cooking, and measurement
  categories that did not lead efficiently to `cat`.

**Changes made:**
1. Capped active hypotheses at 5 per generation. This prevents the guess
   budget from being spread across too many weak categories.
2. Added deduplication: hypotheses that share most of their tried words or
   have near-identical names are merged, keeping the one with the better
   `best_rank`. This prevents mutation from creating redundant categories.
3. Changed the mutation prompt to encourage divergent thinking. Instead of
   only asking for sub-categories, the prompt now asks the LLM to suggest
   at least one genuinely different interpretation of why the best word
   scored well. This should help the solver pivot when it is stuck in the
   wrong semantic area.

