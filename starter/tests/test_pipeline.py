import json
import tempfile
import unittest
from pathlib import Path

from starter.src.chunking import by_section, fixed_size
from starter.src.evaluation import evaluate_query, summarize
from starter.src.ingestion import load_markdown_sections, normalize_text
from starter.src.models import Chunk, DocumentPage
from starter.src.store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]


class IngestionTests(unittest.TestCase):
    def test_normalization_preserves_paragraph_boundaries(self) -> None:
        self.assertEqual(normalize_text("A  \r\n\r\nB\tC"), "A\n\nB C")

    def test_markdown_sections_are_30_traceable_content_units(self) -> None:
        data_dir = ROOT / "data"
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        pages = []
        for document in manifest["documents"]:
            pages.extend(
                load_markdown_sections(
                    data_dir / document["source_path_or_url"],
                    document["source_id"],
                    document["pages"][0]["source"],
                    {"category": document["category"]},
                    document["pages"],
                )
            )
        self.assertEqual(len(pages), 30)
        self.assertTrue(all(len(page.text) >= 1500 for page in pages))


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = DocumentPage(
            document_id="doc",
            source="https://example.test/source",
            page="doc-01",
            text="A" * 1500,
            metadata={"category": "history", "section": "Test"},
        )

    def test_fixed_size_overlap_and_traceability(self) -> None:
        chunks = fixed_size(self.page, size=500, overlap=100)
        self.assertEqual([len(chunk.text) for chunk in chunks], [500, 500, 500, 300])
        self.assertTrue(all(chunk.page == self.page.page for chunk in chunks))
        self.assertEqual(chunks[0].text[-100:], chunks[1].text[:100])

    def test_section_strategy_keeps_short_section_whole(self) -> None:
        chunks = by_section(self.page, max_size=1800)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, self.page.text)
        self.assertEqual(chunks[0].metadata["category"], "history")


class EvaluationTests(unittest.TestCase):
    def test_hit_requires_gold_location_and_complete_quote(self) -> None:
        question = {
            "id": "q",
            "split": "test",
            "gold_source_id": "doc",
            "gold_page": "doc-01",
            "evidence_quote": "verified fact",
        }
        unrelated = {
            "chunk_id": "wrong",
            "text": "verified fact",
            "metadata": {"document_id": "other", "page": "doc-01"},
        }
        correct = {
            "chunk_id": "right",
            "text": "The verified fact is here.",
            "metadata": {"document_id": "doc", "page": "doc-01"},
        }
        scored = evaluate_query(question, [unrelated, correct])
        self.assertFalse(scored["hit_at_1"])
        self.assertTrue(scored["evidence_hit_at_3"])
        self.assertEqual(summarize([{"strategy": "B", **scored}])["B_test"]["n"], 1)


class QdrantStoreTests(unittest.TestCase):
    def test_local_store_filters_and_reingestion_removes_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            store = QdrantVectorStore(Path(directory) / "qdrant", dimensions=2)
            first = Chunk("first", "doc", "source", "doc-01", "history text", {"category": "history"})
            second = Chunk("second", "doc", "source", "doc-02", "news text", {"category": "news"})
            try:
                store.upsert("A", [first, second], [[1.0, 0.0], [0.0, 1.0]])
                history_hits = store.search("A", [1.0, 0.0], filters={"category": "history"})
                self.assertEqual([hit["chunk_id"] for hit in history_hits], ["first"])
                store.upsert("A", [first], [[1.0, 0.0]])
                count = store.client.count(collection_name="hust_A", exact=True).count
                self.assertEqual(count, 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
