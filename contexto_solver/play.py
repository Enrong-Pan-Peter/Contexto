"""Terminal interface for manually playing the local Contexto game."""

from __future__ import annotations

import argparse
import random

from . import config
from .embeddings import EmbeddingModel
from .local_game import LocalGame

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the local Contexto game.")
    parser.add_argument("target", nargs="?", help="Target word. Defaults to a random vocabulary word.")
    parser.add_argument(
        "--embedding-path",
        default=config.GAME_EMBEDDING_PATH,
        help="Path to a text or .npz embedding file for the local game.",
    )
    parser.add_argument("--glove-path", dest="embedding_path", help=argparse.SUPPRESS)
    args = parser.parse_args()

    print("Loading embeddings...")
    model = EmbeddingModel(args.embedding_path)
    target = args.target.lower().strip() if args.target else random.choice(model.vocabulary())
    game = LocalGame(model, target)

    print("Game ready. Guess the word! Type 'quit' to give up.")
    while True:
        guess = input("Guess: ").strip().lower()
        if guess == "quit":
            print(f"The target was {game.get_target()}.")
            return
        if guess == "hint":
            guessed = sorted(
                ((word, rank) for word, rank in game.guesses.items() if rank > 0),
                key=lambda item: item[1],
            )[:5]
            print(f"Best guesses: {guessed}")
            continue

        rank = game.guess(guess)
        if rank == -1:
            print("Word not in vocabulary, try another.")
            continue
        if rank == 1:
            print(f"You got it in {game.total_guesses()} guesses!")
            return

        print(f"Rank: {_color_rank(rank)}{rank}{RESET}")
        best_word, best_rank = game.best_so_far()
        print(f"Best so far: {best_word} ({best_rank}); total guesses: {game.total_guesses()}")


def _color_rank(rank: int) -> str:
    if rank <= 300:
        return GREEN
    if rank <= 1500:
        return YELLOW
    return RED


if __name__ == "__main__":
    main()
