import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

for k in ("OPENAI_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} not set.")


query = "Tell me about GPT-5's architecture."

embeddings = OpenAIEmbeddings(
    model=os.getenv("OPEN_AI_MODEL", "text-embedding-3-small")
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

results = store.similarity_search_with_score(query, k=5)

for i, (doc, score) in enumerate(results, start=1):
    print("=" * 50)
    print(f"Result {i} (score: {score:.2f}):\n{doc.page_content.strip()}\n")

    print("Metadata")
    for k, v in doc.metadata.items():
        print(f"  {k}: {v}")
