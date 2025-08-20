# Platform for Vetting & Recommending Project Topics (ML)

**Project Title:** Design and Implementation of a Platform for Vetting and Recommending Project Topics Using Machine Learning

This service exposes REST endpoints for:
- Storing past projects
- Vetting a new proposed topic (duplicate/similar/novel via cosine-similarity on Sentence-BERT embeddings)
- Basic student profile capture

## Key Endpoints
- `POST /topics` — Add a project {title, introduction, objectives, course?, year?}
- `GET /topics` — List projects
- `POST /vet-topic` — Vet a proposed topic payload {title, introduction, objectives, top_k?, duplicate_threshold?, similar_threshold?}
- `POST /students` — Add student {name, interests[], skills[]}
- `GET /healthz` — Health check

## Running Locally
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Set `.env` or environment variables:
   - `DATABASE_URL=sqlite:///./local.db` (for quick start) or a PostgreSQL URL
   - `EMBEDDING_MODEL=all-MiniLM-L6-v2`
   - `DUPLICATE_THRESHOLD=0.85`
   - `SIMILAR_THRESHOLD=0.70`
4. `uvicorn main:app --reload`
5. Visit `http://localhost:8000/docs`

## Deploy on Render
- Commit and push this repo with the included `render.yaml`.
- Create **New Web Service** → Connect repo → Deploy.
- Render provisions **Postgres** via the YAML and sets `DATABASE_URL` automatically.
- After first deploy, open **Shell** and run: `python seed.py`

## Notes
- Embeddings are L2-normalized and compared with **cosine similarity**.
- FAISS is optional; if import fails, NumPy fallback is used.
- Thresholds can be tuned per request or via env vars.

## References (downloadable online)
- Sentence-BERT (arXiv PDF): https://arxiv.org/pdf/1908.10084
- Sentence-Transformers docs: https://sbert.net/
- Cosine similarity (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- FAISS docs: https://faiss.ai/
- SQLAlchemy 2.0: https://docs.sqlalchemy.org/
- FastAPI docs: https://fastapi.tiangolo.com/
- Render Blueprints: https://render.com/docs/blueprint-spec
