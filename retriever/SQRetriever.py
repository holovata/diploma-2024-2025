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

def SQA_on_abstract(cl)
# Initialize embeddings with SentenceTransformers
print("Initializing SentenceTransformers embeddings...")
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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
