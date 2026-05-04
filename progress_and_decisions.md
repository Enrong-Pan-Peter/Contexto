# Contexto Evolutionary Solver - Progress

Last updated: 2026-05-04


## What we have built

A Python program that solves the Contexto word guessing game using
evolutionary search guided by an LLM (GPT-5.4-mini).

The solver maintains a population of semantic categories. Each generation
it uses the LLM to generate candidate words, evaluates them against the
game (which returns a rank, 1 = correct), selects the best categories,
mutates them into more specific or divergent directions, and optionally
combines categories through crossover. When the best rank drops below a
configurable local-search threshold, the solver asks the LLM for words
semantically close to the current best guess.

All reasoning is logged as human-readable JSON traces so we can inspect
exactly what the solver did and why.

We have two game backends:
- Online: the real Contexto API at api.contexto.me
- Local: our own implementation using GloVe embeddings (glove.6B.300d)


## Current performance

Online game + LLM evolutionary solver:
  Best case ~60 guesses, worst case ~200 guesses after recent fixes in
  the first post-fix verification set. A later API game solved in 254
  guesses, so more targets are needed before reporting a stable range.
  Before fixes worst case was ~500 guesses.

Local GloVe game + LLM evolutionary solver:
  Best case ~60 guesses, worst case ~300-400 guesses after recent fixes.
  Before fixes worst case was 600+ guesses.

Variance across different target words is high.


## Known problems

### 1. Local game performs worse than online game
The LLM solver should be agnostic to the backend embedding model since
it generates words based on semantic reasoning, not vector lookups. But
the local GloVe game consistently produces worse results. Hypothesis:
GPT-5.4-mini's implicit word similarity sense aligns better with the
real Contexto embedding model than with GloVe. Needs more testing with
different embedding models to confirm.

### 2. High variance across games
Some target words are solved in 60 guesses, others take 300 to 500. We
do not yet understand what makes certain words harder. Could be related
to how many distinct semantic neighborhoods the target word belongs to,
or how well the LLM's vocabulary covers that area.

### 3. Poor convergence in some runs
The evolutionary process sometimes stalls. The solver might reach rank 60
early and then barely improve over many generations. The selection and
mutation cycle is not always producing useful new directions.

### 4. Local search gets stuck on moderately close words
When the best rank drops below the local-search threshold, the solver asks
the LLM for words close to the best guess. This works well when the best
word is very close (rank < 5), for example "dog" at rank 2 leads
immediately to "cat" at rank 1. But when the best word is only moderately
close (rank 30 to 100), local search can get stuck, repeatedly guessing
words near the same area without improving.

A better approach would be to give local search a stricter guess budget
and fall back to category exploration if the rank does not improve.

### 5. One-direction problem (partially fixed)
When a word scores well, the solver tends to commit to one interpretation
of that word. For example, "bite" scoring well led the solver entirely
into food categories, missing that "bite" also relates to animals. The
pivot-aware mutation prompt helps but does not fully solve this. The
solver still sometimes goes deep in one direction without branching out.


## What comes next

1. Run more test cases on the local game to check if performance
   stabilizes across different target words.
2. Fix the local search fallback: give local search a guess budget and
   revert to category exploration if it stalls.
3. Look into improving the search strategy without switching to a better
   LLM model. Look at relevant papers for inspiration on
   exploration/exploitation balance and diversity maintenance.
4. Build out the embedding-only evolutionary solver (Stage 3) as a stronger
   baseline. This replaces LLM calls with nearest-neighbor lookups in
   embedding space. Expected to be strong when the embedding model matches
   the game backend, but brittle when it does not. Comparing this against
   the LLM solver will show what the LLM contributes.
5. Try different embedding models for the local game (Word2Vec,
   transformer-based) to test the manifold alignment hypothesis.


## Research framing

The LLM evolutionary solver is a general-purpose approach: it does not
depend on knowing what embedding model the game uses. The embedding-only
solver is a specialist: it should excel when models match but degrade
when they do not. Comparing the two tells us whether LLM semantic
reasoning adds value beyond vector similarity, and how robust each
approach is across different backend models.

The JSON reasoning traces serve as built-in explanations, fitting the
explanation-first XAI paradigm. The search process itself is the
explanation: every category explored, every pivot made, every
selection decision is logged and human-readable.
