from __future__ import annotations
# ruff: noqa: E402

"""
RAG System with optional vector backends and a reliable local fallback.
"""

import json
import importlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings

warnings.filterwarnings("ignore")

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover - pydantic is optional for editor/runtime fallback
    BaseModel = None


LANGCHAIN_AVAILABLE = False
VECTORSTORE_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

ChatOpenAI = None
OpenAIEmbeddings = None
RecursiveCharacterTextSplitter = None
PromptTemplate = None
RetrievalQA = None
FAISS = None
Chroma = None


@dataclass
class LocalDocument:
    page_content: str
    metadata: Dict[str, Any]


DocumentLike = LocalDocument


if BaseModel is not None:
    class MetadataFilter(BaseModel):
        category: Optional[str] = None
        min_accuracy: Optional[float] = None
        timestamp_after: Optional[str] = None
else:
    @dataclass
    class MetadataFilter:
        category: Optional[str] = None
        min_accuracy: Optional[float] = None
        timestamp_after: Optional[str] = None


def _try_load_langchain() -> bool:
    global LANGCHAIN_AVAILABLE
    global ChatOpenAI, OpenAIEmbeddings, RecursiveCharacterTextSplitter
    global PromptTemplate, RetrievalQA

    if LANGCHAIN_AVAILABLE:
        return True

    try:
        _RetrievalQA = importlib.import_module("langchain.chains").RetrievalQA
        # `langchain.prompts` was reorganized across LangChain versions.
        # Try the legacy path first, then fall back to `langchain_core.prompts`.
        try:
            _PromptTemplate = importlib.import_module("langchain.prompts").PromptTemplate
        except Exception:
            _PromptTemplate = importlib.import_module("langchain_core.prompts").PromptTemplate
        langchain_openai = importlib.import_module("langchain_openai")
        _ChatOpenAI = langchain_openai.ChatOpenAI
        _OpenAIEmbeddings = langchain_openai.OpenAIEmbeddings

        try:
            _Splitter = importlib.import_module("langchain_text_splitters").RecursiveCharacterTextSplitter
        except Exception:
            _Splitter = importlib.import_module("langchain.text_splitter").RecursiveCharacterTextSplitter

        ChatOpenAI = _ChatOpenAI
        OpenAIEmbeddings = _OpenAIEmbeddings
        RecursiveCharacterTextSplitter = _Splitter
        PromptTemplate = _PromptTemplate
        RetrievalQA = _RetrievalQA
        LANGCHAIN_AVAILABLE = True
        return True
    except Exception:
        return False


def _try_load_vectorstores() -> bool:
    global VECTORSTORE_AVAILABLE, FAISS, Chroma

    if VECTORSTORE_AVAILABLE:
        return True

    if not _try_load_langchain():
        return False

    try:
        try:
            vectorstores = importlib.import_module("langchain_community.vectorstores")
            _Chroma = vectorstores.Chroma
            _FAISS = vectorstores.FAISS
        except Exception:
            vectorstores = importlib.import_module("langchain.vectorstores")
            _Chroma = vectorstores.Chroma
            _FAISS = vectorstores.FAISS

        FAISS = _FAISS
        Chroma = _Chroma
        VECTORSTORE_AVAILABLE = True
        return True
    except Exception:
        return False


def _try_load_transformers() -> bool:
    global TRANSFORMERS_AVAILABLE

    if TRANSFORMERS_AVAILABLE:
        return True

    try:
        importlib.import_module("sentence_transformers")

        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception:
        return False


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")
    return slug or "document"


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _keyword_score(query: str, document_text: str) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document_text)
    if not query_tokens or not doc_tokens:
        return 0.0

    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    overlap = len(query_set & doc_set)
    if overlap == 0:
        return 0.0

    density = overlap / max(1, len(query_set))
    coverage = overlap / max(1, math.sqrt(len(doc_set)))
    return density + coverage


class RAGSystem:
    """Retrieval-augmented assistant with optional LangChain backends."""

    def __init__(
        self,
        embedding_model: str = "openai",
        vector_store: str = "faiss",
        persist_directory: str = "./vector_db",
        pinecone_api_key: Optional[str] = None,
        pinecone_env: Optional[str] = None,
        pinecone_index: Optional[str] = None,
        auto_bootstrap: bool = True,
    ):
        self.embedding_model_name = embedding_model
        self.vector_store_type = vector_store.lower()
        self.persist_directory = persist_directory
        self.documents: List[DocumentLike] = []
        self.vector_store = None
        self.embeddings = None
        self.llm = None
        self.text_splitter = None
        self._bootstrapped = False
        self._ingested_sources: set[str] = set()
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.pinecone_index_name = pinecone_index

        self._initialize_optional_components()

        if auto_bootstrap:
            self.bootstrap_project_knowledge()

    def _initialize_optional_components(self) -> None:
        if _try_load_langchain() and RecursiveCharacterTextSplitter is not None:
            try:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=900,
                    chunk_overlap=150,
                    length_function=len,
                )
            except Exception:
                self.text_splitter = None

        if self.embedding_model_name == "openai" and OpenAIEmbeddings is not None:
            try:
                self.embeddings = OpenAIEmbeddings()
            except Exception:
                self.embeddings = None

        if ChatOpenAI is not None:
            try:
                self.llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.2)
            except Exception:
                self.llm = None

    def bootstrap_project_knowledge(self, max_files: int = 20) -> int:
        if self._bootstrapped:
            return 0

        root = Path.cwd()
        candidates = [
            root / "README.md",
            root / "QUICKSTART.md",
            root / "INTEGRATION_SUMMARY.md",
            root / "meta_ai_report.txt",
        ]
        candidates.extend(sorted((root / "docs").glob("*.md")))

        loaded_docs: List[str] = []
        loaded_meta: List[Dict[str, Any]] = []

        for path in candidates:
            if len(loaded_docs) >= max_files:
                break
            if not path.exists() or not path.is_file():
                continue
            source_key = str(path.resolve())
            if source_key in self._ingested_sources:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""
            content = _normalize_whitespace(content)
            if not content:
                continue
            loaded_docs.append(content)
            loaded_meta.append(
                {
                    "source": str(path.relative_to(root)).replace("\\", "/"),
                    "type": "documentation",
                    "category": self._infer_category(path.name, content),
                }
            )
            self._ingested_sources.add(source_key)

        if loaded_docs:
            self.ingest_documents(loaded_docs, loaded_meta)

        self._bootstrapped = True
        return len(loaded_docs)

    def _infer_category(self, source_name: str, text: str) -> str:
        source_lower = source_name.lower()
        text_lower = text.lower()
        if "experiment" in text_lower or "metrics" in text_lower:
            return "Experiment"
        if source_lower.endswith(".py") or "code" in text_lower:
            return "Code"
        if "guide" in source_lower or "docs" in source_lower or "api" in text_lower:
            return "Documentation"
        return "General"

    def _split_documents(self, docs: Sequence[DocumentLike]) -> List[DocumentLike]:
        if not docs:
            return []

        if self.text_splitter is not None:
            try:
                split_docs = self.text_splitter.split_documents(list(docs))
                normalized: List[DocumentLike] = []
                for doc in split_docs:
                    normalized.append(
                        LocalDocument(
                            page_content=getattr(doc, "page_content", ""),
                            metadata=dict(getattr(doc, "metadata", {}) or {}),
                        )
                    )
                return normalized
            except Exception:
                return []

        split_docs: List[DocumentLike] = []
        for doc in docs:
            text = _normalize_whitespace(doc.page_content)
            if not text:
                continue
            step = 750
            overlap = 150
            start = 0
            while start < len(text):
                chunk = text[start : start + step]
                meta = dict(doc.metadata)
                meta["chunk_start"] = start
                split_docs.append(LocalDocument(page_content=chunk, metadata=meta))
                if start + step >= len(text):
                    break
                start += step - overlap
        return split_docs

    def _build_vector_store(self, split_docs: Sequence[DocumentLike]) -> None:
        if not split_docs or self.embeddings is None or not _try_load_vectorstores():
            return

        if self.vector_store_type == "faiss" and FAISS is not None:
            try:
                if self.vector_store is None:
                    self.vector_store = FAISS.from_texts(
                        texts=[doc.page_content for doc in split_docs],
                        embedding=self.embeddings,
                        metadatas=[doc.metadata for doc in split_docs],
                    )
                else:
                    self.vector_store.add_texts(
                        texts=[doc.page_content for doc in split_docs],
                        metadatas=[doc.metadata for doc in split_docs],
                    )
            except Exception:
                self.vector_store = None
            return

        if self.vector_store_type == "chromadb" and Chroma is not None:
            try:
                if self.vector_store is None:
                    self.vector_store = Chroma.from_texts(
                        texts=[doc.page_content for doc in split_docs],
                        embedding=self.embeddings,
                        metadatas=[doc.metadata for doc in split_docs],
                        persist_directory=self.persist_directory,
                    )
                else:
                    self.vector_store.add_texts(
                        texts=[doc.page_content for doc in split_docs],
                        metadatas=[doc.metadata for doc in split_docs],
                    )
            except Exception:
                self.vector_store = None

    def ingest_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if metadata is None:
            metadata = [{"source": f"doc_{index}", "category": "General"} for index in range(len(documents))]

        raw_docs = [
            LocalDocument(page_content=document, metadata=dict(meta))
            for document, meta in zip(documents, metadata)
            if document and str(document).strip()
        ]
        split_docs = self._split_documents(raw_docs)
        self.documents.extend(split_docs)
        self._build_vector_store(split_docs)

        if self.vector_store_type == "chromadb" and self.vector_store is not None:
            try:
                self.vector_store.persist()
            except Exception:
                self.vector_store = None

    def ingest_experiment_results(self, experiment_data: Dict[str, Any]) -> None:
        doc_text = self._format_experiment_as_text(experiment_data)
        metadata = {
            "type": "experiment",
            "experiment_id": experiment_data.get("id", "unknown"),
            "timestamp": experiment_data.get("timestamp", ""),
            "models": str(experiment_data.get("models", [])),
            "best_model": experiment_data.get("best_model", ""),
            "best_accuracy": experiment_data.get("best_accuracy", 0.0),
            "category": "Experiment",
            "source": f"experiment::{experiment_data.get('id', 'unknown')}",
        }
        self.ingest_documents([doc_text], [metadata])

    def _format_experiment_as_text(self, experiment_data: Dict[str, Any]) -> str:
        lines = [
            f"Experiment ID: {experiment_data.get('id', 'N/A')}",
            f"Timestamp: {experiment_data.get('timestamp', 'N/A')}",
            f"Dataset: {experiment_data.get('dataset', 'N/A')}",
            f"Target Column: {experiment_data.get('target_column', 'N/A')}",
            f"Models Trained: {', '.join(experiment_data.get('models', []))}",
            f"Best Model: {experiment_data.get('best_model', 'N/A')}",
            "Metrics:",
        ]
        for model_name, metrics in experiment_data.get("metrics", {}).items():
            lines.append(f"{model_name}:")
            for metric, value in metrics.items():
                lines.append(f"  - {metric}: {value}")
        return "\n".join(lines)

    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Optional[Dict[str, Any]]) -> bool:
        if not filter_metadata:
            return True
        for key, expected in filter_metadata.items():
            if metadata.get(key) != expected:
                return False
        return True

    def _local_search(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentLike, float]]:
        scored: List[Tuple[DocumentLike, float]] = []
        for doc in self.documents:
            if not self._matches_filter(doc.metadata, filter_metadata):
                continue
            score = _keyword_score(query, doc.page_content)
            if score <= 0:
                continue
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentLike, float]]:
        if self.vector_store is not None:
            try:
                search_kwargs: Dict[str, Any] = {"k": k}
                if filter_metadata:
                    search_kwargs["filter"] = filter_metadata
                vector_results = self.vector_store.similarity_search_with_score(query, **search_kwargs)
                normalized: List[Tuple[DocumentLike, float]] = []
                for doc, score in vector_results:
                    normalized.append(
                        (
                            LocalDocument(
                                page_content=getattr(doc, "page_content", ""),
                                metadata=dict(getattr(doc, "metadata", {}) or {}),
                            ),
                            float(score),
                        )
                    )
                if normalized:
                    return normalized
            except Exception:
                self.vector_store = None

        return self._local_search(query=query, k=k, filter_metadata=filter_metadata)

    def find_similar_experiments(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        results = self.search(query, k=k, filter_metadata={"type": "experiment"})
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
            }
            for doc, score in results
        ]

    def _invoke_llm(self, prompt: str) -> Optional[str]:
        if self.llm is None:
            return None

        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)
                content = getattr(response, "content", response)
                return str(content).strip()
            if hasattr(self.llm, "predict"):
                return str(self.llm.predict(prompt)).strip()
        except Exception:
            return None

        return None

    def _fallback_answer(self, question: str, sources: Sequence[DocumentLike]) -> str:
        if not sources:
            return "No knowledge base available. Please ingest documents first."

        snippets = []
        for doc in sources[:3]:
            sentence = doc.page_content[:280].strip()
            source_name = doc.metadata.get("source", "unknown")
            snippets.append(f"[{source_name}] {sentence}")

        return (
            f"Based on the retrieved project knowledge, here are the strongest matches for '{question}':\n\n"
            + "\n\n".join(snippets)
        )

    def answer_question(
        self,
        question: str,
        context_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        results = self.search(question, k=context_k, filter_metadata=filter_metadata)
        if not results:
            return {
                "answer": "No knowledge base available. Please ingest documents first.",
                "sources": [],
                "question": question,
            }

        context_docs = [doc for doc, _ in results]
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content[:1200]}"
            for doc in context_docs
        )
        prompt = (
            "You are an expert ML engineer assistant. Use the supplied context only. "
            "If the context is insufficient, say so clearly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        answer = self._invoke_llm(prompt)
        if not answer:
            answer = self._fallback_answer(question, context_docs)

        sources = [
            {
                "content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                "metadata": doc.metadata,
            }
            for doc in context_docs
        ]
        return {"answer": answer, "sources": sources, "question": question}

    def _heuristic_category(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["experiment", "model", "accuracy", "f1", "metrics"]):
            return "Experiment"
        if any(word in query_lower for word in ["code", "function", "class", "python", "api"]):
            return "Code"
        if any(word in query_lower for word in ["guide", "readme", "doc", "documentation", "how to"]):
            return "Documentation"
        return "General"

    def advanced_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        classification_prompt = (
            "Classify the following query into one of: Experiment, Code, Documentation, General. "
            f"Return only the category. Query: {query}"
        )
        category = self._invoke_llm(classification_prompt) or self._heuristic_category(query)
        category = category.strip().splitlines()[0]
        if category not in {"Experiment", "Code", "Documentation", "General"}:
            category = self._heuristic_category(query)

        filter_meta = {"category": category} if category != "General" else None
        result = self.answer_question(query, context_k=k, filter_metadata=filter_meta)
        result["categorization"] = category
        return result

    def recommend_models(self, dataset_description: str, problem_type: str = "classification") -> Dict[str, Any]:
        similar_experiments = self.find_similar_experiments(dataset_description, k=3)
        context = "\n\n".join(
            f"Similar Experiment {index + 1}:\n{item['content']}"
            for index, item in enumerate(similar_experiments)
        )

        prompt = (
            "Based on these similar experiments, recommend the top 3 models to try and explain why.\n\n"
            f"Context:\n{context}\n\n"
            f"Current problem: {dataset_description}\nProblem type: {problem_type}\n"
            "Format as numbered recommendations."
        )
        recommendations = self._invoke_llm(prompt)
        if not recommendations:
            if problem_type == "classification":
                recommendations = (
                    "1. RandomForest - Strong baseline for tabular data and robust to mixed signal.\n"
                    "2. XGBoost - Good when feature interactions and non-linearity matter.\n"
                    "3. LogisticRegression - Useful interpretable baseline for calibration checks."
                )
            else:
                recommendations = (
                    "1. RandomForestRegressor - Solid nonlinear baseline.\n"
                    "2. GradientBoostingRegressor - Good bias/variance tradeoff.\n"
                    "3. LinearRegression - Interpretable baseline for sanity checking."
                )

        return {
            "recommendations": recommendations,
            "similar_experiments": similar_experiments,
            "dataset_description": dataset_description,
        }

    def save(self, path: Optional[str] = None) -> None:
        output_dir = Path(path or self.persist_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "embedding_model": self.embedding_model_name,
            "vector_store": self.vector_store_type,
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ],
        }
        (output_dir / "rag_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if self.vector_store_type == "faiss" and self.vector_store is not None:
            try:
                self.vector_store.save_local(str(output_dir))
            except Exception:
                self.vector_store = None
        elif self.vector_store_type == "chromadb" and self.vector_store is not None:
            try:
                self.vector_store.persist()
            except Exception:
                self.vector_store = None

    def load(self, path: Optional[str] = None) -> None:
        input_dir = Path(path or self.persist_directory)
        manifest_path = input_dir / "rag_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.documents = [
                LocalDocument(page_content=item["page_content"], metadata=item["metadata"])
                for item in payload.get("documents", [])
            ]

        if self.embeddings is None or not _try_load_vectorstores():
            return

        if self.vector_store_type == "faiss" and FAISS is not None:
            try:
                self.vector_store = FAISS.load_local(
                    str(input_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception:
                self.vector_store = None
        elif self.vector_store_type == "chromadb" and Chroma is not None:
            try:
                self.vector_store = Chroma(
                    persist_directory=str(input_dir),
                    embedding_function=self.embeddings,
                )
            except Exception:
                self.vector_store = None


class HybridSearch:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.keyword_index: Dict[str, List[int]] = {}

    def index_keywords(self, documents: List[DocumentLike]) -> None:
        self.keyword_index = {}
        for index, doc in enumerate(documents):
            for word in _tokenize(doc.page_content):
                self.keyword_index.setdefault(word, []).append(index)

    def keyword_search(self, query: str, k: int = 5) -> List[int]:
        scores: Dict[int, int] = {}
        for word in _tokenize(query):
            for doc_index in self.keyword_index.get(word, []):
                scores[doc_index] = scores.get(doc_index, 0) + 1
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_index for doc_index, _ in ranked[:k]]

    def hybrid_search(self, query: str, k: int = 5, semantic_weight: float = 0.7) -> List[DocumentLike]:
        semantic_results = self.rag_system.search(query, k=k)
        keyword_indices = self.keyword_search(query, k=k)
        combined: Dict[int, Dict[str, Any]] = {}

        for doc, score in semantic_results:
            doc_id = id(doc)
            combined[doc_id] = {"doc": doc, "score": float(score) * semantic_weight}

        for doc_index in keyword_indices:
            if doc_index >= len(self.rag_system.documents):
                continue
            doc = self.rag_system.documents[doc_index]
            doc_id = id(doc)
            if doc_id in combined:
                combined[doc_id]["score"] += 1 - semantic_weight
            else:
                combined[doc_id] = {"doc": doc, "score": 1 - semantic_weight}

        ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        return [item["doc"] for item in ranked[:k]]


def create_rag_from_experiments(
    experiments: List[Dict[str, Any]],
    embedding_model: str = "openai",
    vector_store: str = "faiss",
) -> RAGSystem:
    rag = RAGSystem(embedding_model=embedding_model, vector_store=vector_store, auto_bootstrap=False)
    for experiment in experiments:
        rag.ingest_experiment_results(experiment)
    return rag
