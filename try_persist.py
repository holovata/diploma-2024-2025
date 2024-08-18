from langchain_community.vectorstores.chroma import Chroma


db3 = Chroma(collection_name='v_db', persist_directory="./chroma_db", embedding_function=config.embedding_function)

query = "This is a query about AI and data science"
res = db3.similarity_search(query)
print(res)
