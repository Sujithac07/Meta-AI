from pathlib import Path

from llm.rag_system import HybridSearch, RAGSystem


def test_rag_system_local_search_and_answer_fallback():
    rag = RAGSystem(auto_bootstrap=False)
    rag.ingest_documents(
        [
            "RandomForest achieved the best F1 score on the credit risk dataset.",
            "The API guide documents the FastAPI endpoints and request schema.",
        ],
        [
            {"source": "exp-1", "category": "Experiment", "type": "experiment"},
            {"source": "api-docs", "category": "Documentation", "type": "documentation"},
        ],
    )

    result = rag.advanced_query("Which model had the best F1 score?", k=2)

    assert result["categorization"] == "Experiment"
    assert result["sources"]
    assert "RandomForest" in result["answer"]


def test_rag_system_save_and_load_roundtrip(tmp_path: Path):
    rag = RAGSystem(auto_bootstrap=False)
    rag.ingest_documents(
        ["Deployment guide: use uvicorn app.main:app to run the API."],
        [{"source": "deploy", "category": "Documentation"}],
    )

    rag.save(str(tmp_path))

    loaded = RAGSystem(auto_bootstrap=False)
    loaded.load(str(tmp_path))
    result = loaded.search("How do I run the API?", k=1)

    assert len(result) == 1
    assert "uvicorn app.main:app" in result[0][0].page_content


def test_hybrid_search_combines_semantic_and_keyword_results():
    rag = RAGSystem(auto_bootstrap=False)
    rag.ingest_documents(
        [
            "Model governance includes calibration and drift monitoring.",
            "Conversation memory persists session messages in SQLite.",
        ],
        [
            {"source": "governance", "category": "Documentation"},
            {"source": "memory", "category": "Documentation"},
        ],
    )

    hybrid = HybridSearch(rag)
    hybrid.index_keywords(rag.documents)
    results = hybrid.hybrid_search("How is session memory persisted?", k=2)

    assert results
    assert any("SQLite" in doc.page_content for doc in results)
