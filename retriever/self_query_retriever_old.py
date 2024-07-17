from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
import chromadb
from typing import List
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from chromadb.utils import embedding_functions
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Define documents with their metadata
documents = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "science fiction",
            "rating": 9.9,
        },
    ),
]

# Initialize ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Define the custom embeddings class
class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])[0].tolist()

# Create the custom embedding function
embedding_model = CustomEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
'''
# Load documents into Chroma with custom embeddings
vectorStore = Chroma.from_documents(
    documents=documents,
    collection_name="dcd_store",
    embedding=embedding_model,
    persist_directory="../chroma_store2"
)'''



# Define metadata for the retriever
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year_as_integer",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating_as_float", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = Ollama(model="llama3")

# Initialize the vector store with documents and embeddings
vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="dcd_store",
    embedding=embedding_model,
    persist_directory="../chroma_store2"
)
# Debug: Print the documents added to the collection
print("Documents added to the collection:")
for doc in collection.get():
    print(doc)
# Create SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

# Perform the query
query = "What movies are about dinosaurs?"
retrieved_docs = retriever.get_relevant_documents(query)

# Debug: Print retrieved documents
print("Retrieved documents for the query 'What are some movies about dinosaurs?':")
for doc in retrieved_docs:
    print(doc.page_content, doc.metadata)
