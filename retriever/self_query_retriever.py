from langchain_chroma import Chroma
from langchain.llms import Ollama
from langchain_core.documents import Document
import ollama
import chromadb
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

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

# Add documents to the vector store using Ollama embeddings
for i, d in enumerate(documents):
    response = ollama.embeddings(model="nomic-embed-text", prompt=d.page_content)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        metadatas=[d.metadata],
        documents=[d.page_content]
    )

# Debug: Print the documents added to the collection
print("Documents added to the collection:")
for doc in collection.get():
    print(doc)
'''
# Example query
prompt = "What animals are llamas related to?"

# Generate embedding for the query and retrieve the most relevant document
response = ollama.embeddings(
    prompt=prompt,
    model="nomic-embed-text"
)
query_embedding = response["embedding"]
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)
data = results['documents'][0][0]

# Debug: Print the retrieved document
print(f"Most relevant document for the query '{prompt}':")
print(data)
'''
# Define metadata for the retriever
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
llm = Ollama(model="llama3")

# Initialize the vector store with documents
vectorstore = Chroma.from_documents(documents, embeddings=ollama.embeddings)

# Create SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

# Perform the query
query = "What are some movies about dinosaurs?"
retrieved_docs = retriever.retrieve(query)

# Debug: Print retrieved documents
print("Retrieved documents for the query 'What are some movies about dinosaurs?':")
for doc in retrieved_docs:
    print(doc)
