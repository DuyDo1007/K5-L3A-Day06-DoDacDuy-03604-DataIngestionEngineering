"""Ingest the HUST corpus into local Qdrant and run the fixed 20-question benchmark."""
import csv
import hashlib
import json
import platform
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .chunking import by_section, fixed_size
from .embeddings import MultilingualEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_markdown_sections
from .store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_benchmark(questions: list[dict[str, Any]], pages: list[Any]) -> None:
    if len(questions) < 20:
        raise ValueError(f"Benchmark needs at least 20 questions; found {len(questions)}.")
    splits = {split: sum(q["split"] == split for q in questions) for split in ("dev", "test")}
    if splits["dev"] < 15 or splits["test"] < 5 or sum(splits.values()) != len(questions):
        raise ValueError(f"Expected at least 15 dev and 5 test questions; found {splits}.")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Benchmark question IDs must be unique.")
    filtered = [q for q in questions if q.get("filter")]
    if len(filtered) < 3 or not any(q["split"] == "test" for q in filtered):
        raise ValueError("Use at least three metadata-filter questions, including one test question.")
    locations = {(page.document_id, str(page.page)): page for page in pages}
    for question in questions:
        location = (question["gold_source_id"], str(question["gold_page"]))
        if location not in locations:
            raise ValueError(f"{question['id']} points to missing gold page {location}.")
        source_text = locations[location].text.casefold()
        quotes = question.get("evidence_quotes", [question.get("evidence_quote", "")])
        if not any(quote and quote.casefold() in source_text for quote in quotes):
            raise ValueError(f"{question['id']} has no verifiable evidence quote in its gold page.")
        for key, value in question.get("filter", {}).items():
            if key != "source_type" or locations[location].metadata.get(key) != value:
                raise ValueError(f"{question['id']} has a filter inconsistent with its evidence.")
            documents = {page.document_id for page in pages if page.metadata.get(key) == value}
            if len(documents) < 2:
                raise ValueError(f"{question['id']} filter selects fewer than two documents.")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_manual_review(
    path: Path,
    pages: list[Any],
    chunks_by_strategy: dict[str, list[Any]],
    questions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    sample_pages = sorted(pages, key=lambda page: str(page.page))[::3][:10]
    for strategy, chunks in chunks_by_strategy.items():
        for page in sample_pages:
            candidates = [chunk for chunk in chunks if str(chunk.page) == str(page.page)]
            if candidates:
                chunk = candidates[0]
                rows.append(
                    {
                        "review_type": "chunk_coherence",
                        "strategy": strategy,
                        "question_id": "",
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "page": chunk.page,
                        "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
                        "text": chunk.text,
                        "score_0_to_2": "",
                        "metadata_correct": "",
                        "filter_compliant": "",
                        "review_notes": "",
                    }
                )

    test_questions = {question["id"]: question for question in questions if question["split"] == "test"}
    for trace in traces:
        if trace["split"] != "test":
            continue
        question = test_questions[trace["question_id"]]
        top_hits = trace["hits"][:3]
        rows.append(
            {
                "review_type": "grounding",
                "strategy": trace["strategy"],
                "question_id": trace["question_id"],
                "chunk_id": ",".join(hit["chunk_id"] for hit in top_hits),
                "source": "; ".join(hit["metadata"].get("source", "") for hit in top_hits),
                "page": "; ".join(str(hit["metadata"].get("page", "")) for hit in top_hits),
                "metadata": "",
                "text": json.dumps([hit["text"] for hit in top_hits], ensure_ascii=False),
                "score_0_to_2": "",
                "metadata_correct": "",
                "filter_compliant": "",
                "review_notes": f"Question: {question['question']}",
            }
        )
        for hit in top_hits:
            metadata = hit["metadata"]
            filter_data = question.get("filter") or {}
            compliant = (
                all(metadata.get(key) == value for key, value in filter_data.items())
                if filter_data
                else "N/A"
            )
            rows.append(
                {
                    "review_type": "metadata",
                    "strategy": trace["strategy"],
                    "question_id": trace["question_id"],
                    "chunk_id": hit["chunk_id"],
                    "source": metadata.get("source", ""),
                    "page": metadata.get("page", ""),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "text": hit["text"],
                    "score_0_to_2": "",
                    "metadata_correct": "",
                    "filter_compliant": compliant,
                    "review_notes": "",
                }
            )

    fields = [
        "review_type", "strategy", "question_id", "chunk_id", "source", "page", "metadata", "text",
        "score_0_to_2", "metadata_correct", "filter_compliant", "review_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "ingestion.log").open("w", encoding="utf-8") as log:
        try:
            _run(log)
        except Exception as error:
            print(f"ERROR {type(error).__name__}: {error}", file=log)
            raise


def _run(log: Any) -> None:
    benchmark = ROOT / "benchmark"
    config = _read_json(benchmark / "protocol.json")
    inputs = [DATA / "manifest.json", benchmark / "questions.json", benchmark / "protocol.json"]
    manifest = _read_json(inputs[0])
    known_sources = {source["source_path_or_url"] for source in manifest["sources"]}
    pages: list[Any] = []
    for document in manifest["documents"]:
        document_path = DATA / document["source_path_or_url"]
        inputs.append(document_path)
        units = document["pages"]
        if any(not set(unit["source_urls"]) <= known_sources for unit in units):
            raise ValueError(f"Unregistered source URL in {document_path}")
        document_pages = load_markdown_sections(
            document_path,
            document["source_id"],
            units[0]["source"],
            {key: document[key] for key in ("category", "language", "title")}
            | {"source_path": document_path.relative_to(ROOT).as_posix()},
            units,
        )
        pages.extend(document_pages)
        print(f"OK {document_path.relative_to(ROOT).as_posix()}: {len(document_pages)} units", file=log)
    if len(pages) < 30:
        raise ValueError(f"Corpus needs at least 30 section units; found {len(pages)}.")
    short_pages = [(page.page, len(page.text)) for page in pages if len(page.text) < 1500]
    if short_pages:
        raise ValueError(f"Each Markdown content unit must contain at least 1,500 characters: {short_pages}")

    questions = _read_json(inputs[1])
    _validate_benchmark(questions, pages)

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started = time.perf_counter()
    embedder = MultilingualEmbedder(**config["embedding"])

    a, b = config["strategies"]["A"], config["strategies"]["B"]
    strategies = {
        "A": [chunk for page in pages for chunk in fixed_size(page, size=a["chunk_size"], overlap=a["overlap"])],
        "B": [chunk for page in pages for chunk in by_section(page, max_size=b["chunk_size"])],
    }
    strategies["C"] = strategies["B"]
    traces: list[dict[str, Any]] = []
    strategy_stats: dict[str, Any] = {}
    store = QdrantVectorStore(RESULTS / "qdrant_local", embedder.dimensions)
    try:
        for strategy, chunks in strategies.items():
            ingest_started = time.perf_counter()
            texts = [chunk.text for chunk in chunks]
            lengths = embedder.token_counts(texts)
            if max(lengths) > embedder.max_tokens:
                raise ValueError(f"Strategy {strategy}: {max(lengths)} tokens exceeds model limit {embedder.max_tokens}.")
            vectors = embedder.embed_documents(texts)
            store.upsert(strategy, chunks, vectors)
            strategy_stats[strategy] = {
                "chunks": len(chunks),
                "mean_characters": round(sum(len(chunk.text) for chunk in chunks) / len(chunks), 2),
                "min_characters": min(len(chunk.text) for chunk in chunks),
                "max_characters": max(len(chunk.text) for chunk in chunks),
                "total_characters": sum(map(len, texts)),
                "total_tokens": sum(lengths),
                "min_tokens": min(lengths),
                "max_tokens": max(lengths),
                "ingest_seconds": round(time.perf_counter() - ingest_started, 3),
            }

        for split in ("dev", "test"):
            split_questions = [question for question in questions if question["split"] == split]
            query_vectors = embedder.embed_documents([question["question"] for question in split_questions])
            for strategy in strategies:
                for question, vector in zip(split_questions, query_vectors, strict=True):
                    filters = question.get("filter") if strategy == "C" else None
                    hits = store.search(strategy, vector, top_k=config["top_k"], filters=filters)
                    scored = evaluate_query(question, hits)
                    traces.append({"strategy": strategy, "question": question["question"],
                                   "requested_filter": question.get("filter"), "applied_filter": filters,
                                   "gold_source_id": question["gold_source_id"], "gold_page": question["gold_page"],
                                   "evidence_quote": question["evidence_quote"], **scored})
            if split == "dev":
                dev_metrics = summarize(traces)
                selected = max(strategies, key=lambda name: (
                    dev_metrics[f"{name}_dev"]["evidence_hit_at_3"],
                    dev_metrics[f"{name}_dev"]["hit_at_1"],
                    -strategy_stats[name]["chunks"],
                ))
                print(f"DEV_SELECTION {selected} before test queries", file=log, flush=True)
    finally:
        store.close()

    (RESULTS / "traces.jsonl").write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    _write_json(RESULTS / "metrics.json", summarize(traces))
    _write_json(
        RESULTS / "run_metadata.json",
        {
            "corpus_units": len(pages),
            "corpus_characters": sum(len(page.text) for page in pages),
            "questions": len(questions),
            "started_at": started_at,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model": embedder.model_name,
            "model_revision": embedder.revision,
            "embedding_max_tokens": embedder.max_tokens,
            "dimensions": embedder.dimensions,
            "qdrant_client_version": version("qdrant-client"),
            "sentence_transformers_version": version("sentence-transformers"),
            "transformers_version": version("transformers"),
            "torch_version": version("torch"),
            "huggingface_hub_version": version("huggingface-hub"),
            "vector_store": "Qdrant local persistent mode",
            "distance": "cosine",
            "filter": "source_type, applied only in strategy C when the benchmark query provides it",
            "selected_on_dev": selected,
            "inputs_sha256": {
                path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
                for path in inputs
            },
            "total_seconds": round(time.perf_counter() - started, 3),
            "strategies": strategy_stats,
            "cost": "0 API cost; inference runs locally (electricity/hardware not priced)",
        },
    )
    _write_manual_review(RESULTS / "audit_samples.csv", pages, strategies, questions, traces)
    print(f"Ingested {len(pages)} sections, ran {len(questions)} questions across 3 strategies.")
    print(f"Model: {embedder.model_name} ({embedder.dimensions} dimensions)")
    print(f"Results: {RESULTS}")
    print(json.dumps(summarize(traces), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
