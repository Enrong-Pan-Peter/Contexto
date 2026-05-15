"""Build static word-vector caches from sentence-transformer models.

Sentence-transformer checkpoints produce vectors on demand rather than shipping
with a fixed word list like GloVe. This tool encodes a chosen vocabulary once so
the local game and embedding solver can keep using the existing EmbeddingModel
interface.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"


def main() -> None:
    args = _parse_args()
    words = _load_vocabulary(
        Path(args.vocab_source),
        limit=args.limit,
        lowercase=args.lowercase,
        alphabetic_only=args.alphabetic_only,
        min_length=args.min_length,
    )
    if not words:
        raise ValueError("No vocabulary words available after filtering.")

    model = _load_sentence_transformer(args.model, args.device)
    texts = [args.input_template.format(word=word) for word in words]
    vectors = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
        show_progress_bar=True,
    ).astype(np.float32)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_model": args.model,
        "vocab_source": str(args.vocab_source),
        "input_template": args.input_template,
        "word_count": len(words),
        "dimension": int(vectors.shape[1]),
        "normalized": args.normalize,
        "lowercase": args.lowercase,
        "alphabetic_only": args.alphabetic_only,
        "min_length": args.min_length,
        "limit": args.limit,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "text":
        _write_text(output_path, words, vectors)
    else:
        _write_npz(output_path, words, vectors, metadata)

    print(f"Wrote {len(words):,} embeddings to {output_path}")
    print(json.dumps(metadata, indent=2))


def _load_sentence_transformer(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to build transformer embedding caches. "
            "Install project requirements with `python -m pip install -r requirements.txt`."
        ) from exc

    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def _load_vocabulary(
    path: Path,
    limit: int | None,
    lowercase: bool,
    alphabetic_only: bool,
    min_length: int,
) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for raw_word in _iter_vocabulary_words(path):
        word = raw_word.strip()
        if lowercase:
            word = word.lower()
        if not word or word in seen:
            continue
        if len(word) < min_length:
            continue
        if alphabetic_only and re.fullmatch(r"[a-z]+", word) is None:
            continue
        seen.add(word)
        words.append(word)
        if limit is not None and len(words) >= limit:
            break
    return words


def _iter_vocabulary_words(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as vocab_file:
        for line in vocab_file:
            stripped = line.strip()
            if not stripped:
                continue
            # GloVe-style files use the first whitespace-delimited field as the word.
            yield stripped.split()[0]


def _write_text(path: Path, words: list[str], vectors: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as output:
        for word, vector in zip(words, vectors):
            values = " ".join(f"{float(value):.8g}" for value in vector)
            output.write(f"{word} {values}\n")


def _write_npz(path: Path, words: list[str], vectors: np.ndarray, metadata: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        words=np.asarray(words, dtype=str),
        vectors=vectors.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static word embedding cache.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MINILM_MODEL,
        help=f"Sentence-transformers model name. Default: {DEFAULT_MINILM_MODEL}",
    )
    parser.add_argument("--vocab-source", required=True, help="Vocabulary file or GloVe-style embedding file.")
    parser.add_argument("--output", required=True, help="Output .npz or text-vector path.")
    parser.add_argument("--format", choices=["npz", "text"], default="npz", help="Cache output format.")
    parser.add_argument("--limit", type=int, help="Maximum number of vocabulary words to encode.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", help="Optional sentence-transformers device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--input-template", default="{word}", help="Text template passed to the encoder.")
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    parser.add_argument("--allow-nonalphabetic", action="store_false", dest="alphabetic_only")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.set_defaults(lowercase=True, alphabetic_only=True, normalize=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
