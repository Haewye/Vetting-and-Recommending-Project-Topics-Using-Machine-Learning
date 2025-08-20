from sqlalchemy.orm import Session
from main import SessionLocal, Project, rebuild_faiss_index

def seed():
    with SessionLocal() as db:
        samples = [
            ("An Intelligent System for Project Topic Recommendation using NLP", 
             "We propose a system that recommends topics to students based on interests and prior works.",
             "Build a sentence-embedding index; compute cosine similarity; provide suggestions."),
            ("A Platform for Vetting Student Project Topics with Sentence-BERT",
             "Platform checks for duplicates by comparing semantic embeddings of title, intro and objectives.",
             "Reduce duplication; assist supervisors; ensure novelty."),
            ("Semantic Search of Undergraduate Project Repositories",
             "We implement vector search to find conceptually related projects across years.",
             "Index all projects with FAISS; implement nearest-neighbor queries."),
        ]
        for t, i, o in samples:
            db.add(Project(title=t, introduction=i, objectives=o, course="Computer Science", year=2024))
        db.commit()
        rebuild_faiss_index(db)
        print("Seeded sample projects.")

if __name__ == "__main__":
    seed()
