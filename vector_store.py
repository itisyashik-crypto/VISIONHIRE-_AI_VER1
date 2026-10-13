from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

from config import AppConfig


class VectorStoreManager:
    def __init__(self, knowledge_base_dir: str | Path | None = None):
        self.knowledge_base_dir = Path(knowledge_base_dir or AppConfig.KNOWLEDGE_BASE_PATH)
        self.vector_store = None
        self.embeddings = None
        self.documents: List[Document] = []
        self.index_path: Path | None = None
        self._initialize()

    def _initialize(self):
        source_documents = self._load_documents()
        if not source_documents:
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True,
        )
        self.documents = splitter.split_documents(source_documents)
        content_signature = hashlib.sha256(
            "\n".join(
                f"{doc.metadata.get('source', '')}:{doc.page_content}"
                for doc in self.documents
            ).encode("utf-8")
        ).hexdigest()
        self.index_path = AppConfig.VECTOR_DB_PATH / content_signature

        if not AppConfig.GOOGLE_API_KEY:
            return
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=AppConfig.GOOGLE_API_KEY,
            )
            index_file = self.index_path / "index.faiss"
            manifest_file = self.index_path / "manifest.json"
            if index_file.exists() and manifest_file.exists():
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if manifest.get("content_signature") == content_signature:
                    self.vector_store = FAISS.load_local(
                        str(self.index_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    return

            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_path))
            manifest_file.write_text(
                json.dumps({"content_signature": content_signature}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            self.embeddings = None
            self.vector_store = None

    def _extract_pdf_text(self, file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            chunks = []
            for page in reader.pages:
                text = page.extract_text() or ""
                chunks.append(text)
            return "\n".join(chunks)
        except Exception:
            return ""

    def _load_documents(self) -> List[Document]:
        docs: List[Document] = []
        if not self.knowledge_base_dir.exists():
            return docs
        for file in sorted(self.knowledge_base_dir.iterdir()):
            if (
                not file.is_file()
                or file.name.lower() == "readme.md"
                or file.suffix.lower() not in {".md", ".txt", ".pdf"}
            ):
                continue
            try:
                text = ""
                if file.suffix.lower() == ".pdf":
                    text = self._extract_pdf_text(file)
                else:
                    text = file.read_text(encoding="utf-8", errors="ignore")

                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file.name},
                    ))
            except Exception:
                continue
        return docs

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        tokens = set(re.findall(r"[a-z0-9+#.-]+", query.lower()))
        stop_words = {"the", "and", "for", "with", "from", "that", "this", "into", "based", "role"}
        tokens -= stop_words
        if not tokens:
            return []

        ranked = []
        for document in self.documents:
            document_tokens = set(re.findall(r"[a-z0-9+#.-]+", document.page_content.lower()))
            overlap = tokens & document_tokens
            if not overlap:
                continue
            relevance = len(overlap) / len(tokens)
            ranked.append((relevance, document))
        ranked.sort(key=lambda item: (item[0], len(item[1].page_content)), reverse=True)
        return [
            {
                "content": document.page_content,
                "score": round(float(score), 4),
                "source": document.metadata.get("source", "knowledge_base"),
                "retrieval_method": "keyword_fallback",
            }
            for score, document in ranked[:k]
        ]

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return self._keyword_search(query, k)

        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": round(float(score), 4),
                "source": doc.metadata.get("source", "knowledge_base"),
                "retrieval_method": "faiss",
            }
            for doc, score in results
        ]
