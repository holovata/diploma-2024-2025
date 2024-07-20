from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain_core.embeddings.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings
import uuid

# Initialize embeddings with SentenceTransformers
print("Initializing SentenceTransformers embeddings...")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create documents
docs = [
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
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

# Convert documents to embeddings using SentenceTransformers
texts = [doc.page_content for doc in docs]
# texts_emb = embeddings.embed_documents(texts)
'''embeddings = model.encode(texts)
for i, doc in enumerate(docs):
    doc.embedding = embeddings[i]'''
print("Embeddings initialized.")
# Initialize the vector store with SentenceTransformers embeddings
print("Initializing vector store...")
vectorstore = Chroma.from_documents(docs, embedding_function)
print("Vector store initialized.")

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"

# Initialize the language model
print("Initializing language model...")
llm = Ollama(model="phi3:medium")
print("Language model initialized.")

# Initialize the retriever
print("Initializing retriever...")
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
print("Retriever initialized.")

# This example only specifies a relevant query
print("Invoking retriever...")
response = retriever.invoke("What are some movies about dinosaurs")
print("Retriever invoked.")
print(response)
