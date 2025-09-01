# app.py
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Vector / NLP
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# SQLAlchemy (SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Analytics / Utils
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer

# ===================== Paths & Constants =====================
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"
DB_PATH = DATA_DIR / "docs.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 5

# ===================== Database Models =====================
class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    size_kb = Column(Float, default=0.0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ===================== Embeddings / Vectorstore =====================
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
vectorstore: Optional[FAISS] = None  # in-memory handle

def load_or_create_index() -> Optional[FAISS]:
    idx = INDEX_DIR / "index.faiss"
    pkl = INDEX_DIR / "index.pkl"
    if idx.exists() and pkl.exists():
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return None

def save_index(vs: FAISS) -> None:
    vs.save_local(str(INDEX_DIR))

def ensure_vectorstore() -> None:
    """Load index if present, else create a new empty one (cosine-normalized)."""
    global vectorstore
    if vectorstore is None:
        v = load_or_create_index()
        if v is None:
            vectorstore = FAISS.from_texts(
                texts=["(bootstrap)"],
                embedding=embeddings,
                metadatas=[{"source": "system"}],
                normalize_L2=True,  # cosine-style
            )
            save_index(vectorstore)
        else:
            vectorstore = v

# ===================== File Helpers =====================
def pdf_to_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)

def txt_to_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def clean_text(text: str, *, lowercase: bool = False, drop_references: bool = False) -> str:
    """Lightweight cleaner for better embeddings."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # remove page headers/footers like "Page 3" or bare page numbers
    t = re.sub(r'^\s*(page\s+\d+(\s+of\s+\d+)?)\s*$', "", t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'^\s*\d+\s*$', "", t, flags=re.MULTILINE)
    # fix hyphen line breaks: "some-\nthing" -> "something"
    t = re.sub(r'(\w)-\n(\w)', r"\1\2", t)
    # merge single line breaks within paragraphs
    t = re.sub(r'(?<!\n)\n(?!\n)', " ", t)
    # strip URLs/emails noise
    t = re.sub(r'(https?://\S|www\.)\S+', "", t)
    t = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', "", t)
    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if drop_references:
        parts = re.split(r'\n\s*references\s*\n', t, flags=re.IGNORECASE)
        t = parts[0]
    if lowercase:
        t = t.lower()
    return t.strip()

def chunk_text(text: str, source_name: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text], metadatas=[{"source": source_name}])

def read_all_uploaded_texts(clean: bool = True) -> List[str]:
    texts: List[str] = []
    for p in UPLOAD_DIR.glob("*"):
        try:
            if p.suffix.lower() == ".pdf":
                t = pdf_to_text(p)
            elif p.suffix.lower() == ".txt":
                t = txt_to_text(p)
            else:
                continue
            texts.append(clean_text(t) if clean else t)
        except Exception:
            pass
    return texts

# ===================== Retrieval Helpers =====================
def extract_best_sentences(snippet: str, query: str, n: int = 2) -> str:
    """Extract up to n sentences from snippet that best match the query words."""
    sentences = re.split(r'(?<=[.!?])\s+', snippet.strip())
    if not sentences:
        return snippet.strip()
    qwords = set(query.lower().split())
    scored = [(sum(1 for w in qwords if w in s.lower()), s) for s in sentences]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:n]]
    return " ".join(top).strip()

@dataclass
class Result:
    page_content: str
    metadata: dict
    score: float  # 0..1 (we map distance → similarity)

# ===================== Flask App =====================
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    ensure_vectorstore()
    with SessionLocal() as s:
        recent = s.query(Document).order_by(Document.uploaded_at.desc()).limit(10).all()
    return render_template("index.html", results=None, message=None, recent=recent)

@app.route("/upload", methods=["POST"])
def upload():
    ensure_vectorstore()
    files = request.files.getlist("files")
    added_files = 0
    total_chunks = 0
    all_chunks: List[str] = []
    all_meta: List[dict] = []

    with SessionLocal() as s:
        for f in files:
            if not f.filename:
                continue
            name = secure_filename(f.filename)
            dest = UPLOAD_DIR / name
            f.save(dest)

            ext = dest.suffix.lower()
            if ext == ".pdf":
                text = clean_text(pdf_to_text(dest))
            elif ext == ".txt":
                text = clean_text(txt_to_text(dest))
            else:
                continue

            # record metadata
            stat = dest.stat()
            s.add(Document(
                filename=name,
                size_kb=round(stat.st_size / 1024.0, 2),
                uploaded_at=datetime.utcnow(),
            ))

            # chunk & collect
            docs = chunk_text(text, source_name=name)
            all_chunks.extend([d.page_content for d in docs])
            all_meta.extend([d.metadata for d in docs])
            added_files += 1
            total_chunks += len(docs)

        s.commit()

    if all_chunks:
        global vectorstore
        vectorstore.add_texts(all_chunks, metadatas=all_meta)  # normalized index already
        save_index(vectorstore)

    msg = f"Indexed {added_files} file(s), {total_chunks} chunk(s)."
    with SessionLocal() as s:
        recent = s.query(Document).order_by(Document.uploaded_at.desc()).limit(10).all()
    return render_template("index.html", results=None, message=msg, recent=recent)

@app.route("/search", methods=["GET"])
def search():
    ensure_vectorstore()
    q = request.args.get("q", "").strip()
    if not q:
        return redirect(url_for("home"))

    # Retrieve top-k (distance smaller = better). You can switch to MMR if desired.
    hits = vectorstore.similarity_search_with_score(q, k=TOP_K)

    results: List[Result] = []
    for doc, dist in hits:
        if doc.page_content.strip() == "(bootstrap)":
            continue
        sim = 1.0 / (1.0 + float(dist))  # map distance → similarity (0..1)
        snippet = extract_best_sentences(doc.page_content, q, n=2)
        results.append(Result(snippet, doc.metadata, sim))

    with SessionLocal() as s:
        recent = s.query(Document).order_by(Document.uploaded_at.desc()).limit(10).all()

    return render_template("index.html", results=results, message=None, recent=recent)

@app.route("/analytics", methods=["GET"])
def analytics():
    ensure_vectorstore()
    corpus = read_all_uploaded_texts(clean=True)
    top_terms = []
    if corpus:
        cv = CountVectorizer(stop_words="english", max_features=5000, ngram_range=(1,1))
        X = cv.fit_transform(corpus)
        freqs = X.toarray().sum(axis=0)
        vocab = cv.get_feature_names_out()
        pairs = list(zip(vocab, freqs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_terms = pairs[:10]

    with SessionLocal() as s:
        recent = s.query(Document).order_by(Document.uploaded_at.desc()).limit(10).all()

    labels = [t for t, _ in top_terms]
    values = [int(c) for _, c in top_terms]
    return render_template("index.html",
                           results=None,
                           message=None,
                           recent=recent,
                           chart_labels=labels,
                           chart_values=values)

@app.route("/delete/<int:doc_id>", methods=["POST"])
def delete(doc_id: int):
    ensure_vectorstore()
    with SessionLocal() as s:
        doc = s.query(Document).filter_by(id=doc_id).first()
        if not doc:
            return redirect(url_for("home"))

        file_path = UPLOAD_DIR / doc.filename
        if file_path.exists():
            file_path.unlink()

        s.delete(doc)
        s.commit()

    # Rebuild FAISS index (normalized) from whatever files remain
    new_docs = []
    for p in UPLOAD_DIR.glob("*"):
        try:
            if p.suffix.lower() == ".pdf":
                text = clean_text(pdf_to_text(p))
            elif p.suffix.lower() == ".txt":
                text = clean_text(txt_to_text(p))
            else:
                continue
            new_docs.extend(chunk_text(text, source_name=p.name))
        except Exception:
            pass

    global vectorstore
    if new_docs:
        vectorstore = FAISS.from_documents(new_docs, embeddings, normalize_L2=True)
    else:
        vectorstore = FAISS.from_texts(
            texts=["(bootstrap)"],
            embedding=embeddings,
            metadatas=[{"source": "system"}],
            normalize_L2=True,
        )
    save_index(vectorstore)
    return redirect(url_for("home"))

# ===================== Entrypoint =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
