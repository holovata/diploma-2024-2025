from langchain_community.vectorstores.chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from CONFIG import config

db3 = Chroma(persist_directory="./chroma_store", embedding_function=config.embedding_function)

'''documents = db3.get()['documents']
print(f"Number of documents in vector store: {len(documents)}")
if len(documents) > 0:
    print("Sample document:", documents[0])'''

print(db3._collection.count())