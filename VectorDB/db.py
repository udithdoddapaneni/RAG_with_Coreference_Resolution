"""Run from inside VectorDB directory"""
from models import Documents, Query
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import uuid
import fastapi
import toml
import uvicorn

app = fastapi.FastAPI()
config = toml.load("config.toml")
model = HuggingFaceEmbeddings(**config["embedder"])

class ChromaDB:
    """Class for chroma DB"""
    def __init__(self):
        self.path = "ChromaDB/"
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection("vectorspace", metadata={"hnsw:space":"cosine"})
    def add(self, documents:list[str], filenames:list[str], embeddings:list[float]):
        self.collection.add(
            ids=[str(uuid.uuid4()) for i in documents],
            embeddings=embeddings,
            documents=documents,
            metadatas=[{"filename":filename} for filename in filenames],
        )
    def reset(self):
        self.client.delete_collection("vectorspace")
        self.collection = self.client.get_or_create_collection("vectorspace", metadata={"hnsw:space":"cosine"})
    def query(self, embedding:list[float], n:int):
        return self.collection.query(
            query_embeddings=embedding,
            n_results=n
        )
DB = ChromaDB()

@app.post("/save")
async def save(documents:Documents) -> dict:
    """Save documents in the vector database"""
    try:
        embeddings = await model.aembed_documents(documents.documents)
        DB.add(
            documents=documents.documents,
            filenames=documents.filenames,
            embeddings=embeddings
        )
        return {"response": "success"}
    except Exception as e:
        return {"response": "failure", "exception": str(e)}
    
@app.get("/reset")
async def reset() -> dict:
    """Reset the vector database"""
    try:
        DB.reset()
        return {"response": "success"}
    except Exception as e:
        return {"response": "failure", "exception": str(e)}
    
@app.post("/retrieve")
async def retrieve(query:Query) -> dict:
    """Retrieve most similar documents"""
    try:
        embedding = await model.aembed_query(query.query)
        documents = DB.query(embedding=embedding, n=query.n)["documents"]
        return {"response": "success", "documents": documents}
    except Exception as e:
        return {"response": "failure", "exception": str(e)}
    

if __name__ == "__main__":
    uvicorn.run("__main__:app", **config["service"])