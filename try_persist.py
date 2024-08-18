from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db3 = Chroma(persist_directory="./chroma_store", embedding_function=embedding_function)

documents = db3.get()['documents']
print(f"Number of documents in vector store: {len(documents)}")
if len(documents) > 0:
    print("Sample document:", documents[0])