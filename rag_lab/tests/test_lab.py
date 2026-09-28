from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_lab.chunking import chunk_documents
from rag_lab.corpus import load_documents
from rag_lab.embeddings import TfidfEmbedder, cosine_similarity
from rag_lab.evaluation import evaluate, load_questions
from rag_lab.metrics import retrieval_metrics
from rag_lab.retrieval import Retriever


class DatasetTests(unittest.TestCase):
    def test_corpus_and_splits_are_complete(self) -> None:
        documents = load_documents(ROOT / "documents")
        dev = load_questions(ROOT / "data" / "dev_questions.json")
        public_test = load_questions(ROOT / "data" / "test_questions.json", require_gold=False)
        self.assertEqual(len(documents), 12)
        self.assertEqual(len(dev), 30)
        self.assertEqual(len(public_test), 10)
        document_ids = {document.document_id for document in documents}
        for item in dev:
            self.assertTrue(set(item["relevant_doc_ids"]) <= document_ids)

        gold_path = ROOT / "instructor" / "test_gold.json"
        if gold_path.is_file():
            gold = load_questions(gold_path)
            self.assertEqual(
                [(item["id"], item["question"]) for item in public_test],
                [(item["id"], item["question"]) for item in gold],
            )
            for item in gold:
                self.assertTrue(set(item["relevant_doc_ids"]) <= document_ids)

    def test_public_test_has_no_labels_or_answers(self) -> None:
        raw = json.loads((ROOT / "data" / "test_questions.json").read_text(encoding="utf-8"))
        for item in raw:
            self.assertNotIn("relevant_doc_ids", item)
            self.assertNotIn("expected_answer", item)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        documents = load_documents(ROOT / "documents")
        chunks = chunk_documents(documents, chunk_size=120, overlap=20)
        self.retriever = Retriever(chunks, TfidfEmbedder())

    def test_retrieves_document_lifecycle(self) -> None:
        results = self.retriever.retrieve("Сколько секунд действует ссылка на исходник?", top_k=5)
        self.assertIn("doc_04", [result.chunk.document_id for result in results])

    def test_dev_evaluation_returns_all_metrics(self) -> None:
        questions = load_questions(ROOT / "data" / "dev_questions.json")
        report = evaluate(self.retriever, questions, ks=[1, 3, 5, 10])
        self.assertEqual(report["question_count"], 30)
        self.assertEqual(set(report["aggregate"]), {"1", "3", "5", "10"})
        self.assertGreaterEqual(report["aggregate"]["5"]["hit"], 0.7)


class MetricTests(unittest.TestCase):
    def test_metrics_use_first_relevant_rank(self) -> None:
        result = retrieval_metrics(["doc_a", "doc_b", "doc_c"], {"doc_b"}, k=3)
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["hit"], 1.0)
        self.assertEqual(result["reciprocal_rank"], 0.5)

    def test_cosine(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
