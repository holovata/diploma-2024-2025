from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# Define custom embeddings class
class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()

# Load and split the document
loader = PyPDFLoader("../example_pdfs/opytuvannya.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create the custom embedding function
embedding_model = CustomEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load it into Chroma
vectorStore = Chroma.from_documents(
    documents=docs,
    collection_name="dcd_store",
    embedding=embedding_model,
    persist_directory="../chroma_store"  # Specify the persist directory here
)

# Query it
query = "Яке ставлення до непрофесійної політичної діяльності в Україні?"
results = vectorStore.similarity_search(query)

# Print results
if results:
    for result in results:
        print(result.page_content)
else:
    print("No relevant results found.")
