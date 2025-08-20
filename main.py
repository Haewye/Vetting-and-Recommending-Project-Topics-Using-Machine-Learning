import os
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, select, String, Float, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

from sentence_transformers import SentenceTransformer, util
import numpy as np

# Optional FAISS acceleration
USE_FAISS = False
try:
    import faiss  # type: ignore
    USE_FAISS = True
except Exception:
    USE_FAISS = False

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.85))
SIMILAR_THRESHOLD = float(os.getenv("SIMILAR_THRESHOLD", 0.70))

# ---------- DB setup ----------
class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(300))
    introduction: Mapped[str] = mapped_column(Text)
    objectives: Mapped[str] = mapped_column(Text)
    course: Mapped[Optional[str]] = mapped_column(String(120), default=None)
    year: Mapped[Optional[int]] = mapped_column(Integer, default=None)

class Student(Base):
    __tablename__ = "students"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120))
    interests: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON list
    skills: Mapped[Optional[str]] = mapped_column(Text, default=None)     # JSON list

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base.metadata.create_all(engine)

# ---------- ML setup ----------
model = SentenceTransformer(EMBEDDING_MODEL)

def project_text_blob(title: str, introduction: str, objectives: str) -> str:
    return f"Title: {title}\nIntroduction: {introduction}\nObjectives: {objectives}"

def embed_texts(texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return emb

# In-memory FAISS index for speed
FAISS_INDEX = None
PROJECT_ID_TO_VECIDX: Dict[int, int] = {}

def rebuild_faiss_index(db: Session):
    global FAISS_INDEX, PROJECT_ID_TO_VECIDX
    PROJS = db.execute(select(Project)).scalars().all()
    blobs = [project_text_blob(p.title, p.introduction, p.objectives) for p in PROJS]
    if not blobs:
        FAISS_INDEX = None
        PROJECT_ID_TO_VECIDX = {}
        return
    embs = embed_texts(blobs)
    if USE_FAISS:
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs.astype(np.float32))
        FAISS_INDEX = index
    else:
        FAISS_INDEX = embs  # numpy fallback
    PROJECT_ID_TO_VECIDX = {p.id: i for i, p in enumerate(PROJS)}

def cosine_scores(query_emb: np.ndarray, db_embs: np.ndarray, top_k: int = 10):
    scores = (query_emb @ db_embs.T).flatten()
    idx = np.argsort(-scores)[:top_k]
    return idx, scores[idx]

# ---------- Schemas ----------
class ProjectIn(BaseModel):
    title: str = Field(..., max_length=300)
    introduction: str
    objectives: str
    course: Optional[str] = None
    year: Optional[int] = None

class ProjectOut(ProjectIn):
    id: int

class VetRequest(BaseModel):
    title: str
    introduction: str
    objectives: str
    top_k: int = 5
    duplicate_threshold: Optional[float] = None
    similar_threshold: Optional[float] = None

class StudentIn(BaseModel):
    name: str
    interests: List[str] = []
    skills: List[str] = []

class StudentOut(BaseModel):
    id: int
    name: str
    interests: List[str] = []
    skills: List[str] = []

# ---------- FastAPI ----------
app = FastAPI(
    title="Project Topic Vetting & Recommendation API",
    description="Design and Implementation of a Platform for Vetting and Recommending Project Topics Using Machine Learning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup():
    with SessionLocal() as db:
        rebuild_faiss_index(db)

# ---------- Endpoints ----------
@app.post("/topics", response_model=ProjectOut)
def add_topic(payload: ProjectIn, db: Session = Depends(get_db)):
    pr = Project(**payload.model_dump())
    db.add(pr)
    db.commit()
    db.refresh(pr)
    rebuild_faiss_index(db)
    return ProjectOut(id=pr.id, **payload.model_dump())

@app.get("/topics", response_model=List[ProjectOut])
def list_topics(db: Session = Depends(get_db)):
    rows = db.execute(select(Project)).scalars().all()
    return [ProjectOut(id=r.id, title=r.title, introduction=r.introduction, objectives=r.objectives, course=r.course, year=r.year) for r in rows]

@app.post("/students", response_model=StudentOut)
def add_student(payload: StudentIn, db: Session = Depends(get_db)):
    st = Student(name=payload.name, interests=json.dumps(payload.interests), skills=json.dumps(payload.skills))
    db.add(st)
    db.commit()
    db.refresh(st)
    return StudentOut(id=st.id, name=st.name, interests=payload.interests, skills=payload.skills)

@app.post("/vet-topic")
def vet_topic(req: VetRequest, db: Session = Depends(get_db)):
    dup_th = req.duplicate_threshold or DUPLICATE_THRESHOLD
    sim_th = req.similar_threshold or SIMILAR_THRESHOLD

    blob = project_text_blob(req.title, req.introduction, req.objectives)
    q = embed_texts([blob]).astype(np.float32)

    projs = db.execute(select(Project)).scalars().all()
    if not projs:
        return {"status": "OK", "matches": [], "notes": "No projects in database yet."}

    # Search
    if USE_FAISS and not isinstance(FAISS_INDEX, np.ndarray):
        D, I = FAISS_INDEX.search(q, min(req.top_k, len(projs)))
        sims = D[0]
        idxs = I[0]
        results = [(projs[i], float(sims[j])) for j, i in enumerate(idxs)]
    else:
        db_embs = FAISS_INDEX  # numpy matrix
        idx, scores = cosine_scores(q, db_embs, top_k=min(req.top_k, len(projs)))
        results = [(projs[i], float(scores[j])) for j, i in enumerate(idx)]

    def label_for(score: float) -> str:
        if score >= dup_th:
            return "DUPLICATE"
        if score >= sim_th:
            return "SIMILAR"
        return "NOVEL"

    matches = [{
        "project_id": p.id,
        "title": p.title,
        "similarity": round(score, 4),
        "label": label_for(score)
    } for (p, score) in results]

    # Suggestion heuristics: tweak method/dataset/domain when duplicate/similar
    suggestions = []
    for m in matches:
        if m["label"] in {"DUPLICATE", "SIMILAR"}:
            suggestions.append(f"Focus on a different method or dataset for: '{m['title']}'. For example, add a new domain (e.g., healthcare, agriculture) or a technique (e.g., graph neural networks, topic modeling).")

    if not suggestions:
        suggestions.append("Proceed. Topic appears sufficiently novel under current thresholds.")

    return {"status": "OK", "matches": matches, "thresholds": {"duplicate": dup_th, "similar": sim_th}, "suggestions": suggestions}

@app.get("/healthz")
def health():
    return {"ok": True}
