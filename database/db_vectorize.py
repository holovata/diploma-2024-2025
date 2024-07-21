# database/db_vectorize.py
import ollama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.llms import Ollama
import chromadb
import uuid
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from database.db_select import get_all_papers


def create_chroma_index():
    # Инициализация ChromaDB клиента
    # client = chromadb.Client()
    collection_name = 'papers_collection'
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    '''# Проверка существования коллекции
    if collection_name in client.list_collections():
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(collection_name)'''

    # collection = client.get_or_create_collection(collection_name)
    # Загрузка предобученной модели для преобразования текстов в векторы
    # model = Ollama(model="nomic-embed-text")

    # Извлечение данных из базы данных
    papers = get_all_papers()

    # Преобразование аннотаций в векторы
    texts = [paper[4] for paper in papers]  # Аннотации статей
    # print("Abstracts:")
    # print(texts)
    # print(len(texts))
    # vectors = ollama.embeddings(model="nomic-embed-text", prompt=texts)

    # Создание списков для ID, векторов и метаданных
    ids = [str(uuid.uuid4()) for _ in papers]
    # embeddings = vectors.tolist()
    metadatas = [{
        'name': paper[1],
        'authors': paper[2],
        'url': paper[3],
        'abstract': paper[4],  # Сохранение аннотации в метаданных
        'keyword': paper[5],
        'categories': paper[6],
        'year': paper[7],
        'eprint': paper[8]
    } for paper in papers]

    # Добавление данных в коллекцию
    # collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print("Initializing vector store...")
    vectorstore = Chroma.from_texts(texts=texts, embedding=embedding_function,
                                    collection_name=collection_name,metadatas=metadatas,
                                    ids=ids)
    print("Vector store initialized.")
    # print("Vector index has been created and stored in ChromaDB.")
    # print(collection.count())
    print(len(vectorstore.get()['documents']))
    # return client, collection, texts


# create_chroma_index()


'''def search_chroma_index(query, top_k=5):
    # Инициализация ChromaDB клиента
    # client = chromadb.Client()
    # collection = client.get_collection('papers_collection')
    client, collection, papers = create_chroma_index()
    # Загрузка предобученной модели для преобразования текстов в векторы
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Преобразование запроса в вектор
    # query_vector = model.encode([query])

    # Поиск ближайших соседей
    response = collection.query(query_texts=query, n_results=top_k)

    # Возвращение результатов поиска
    if 'metadatas' in response and 'distances' in response:
        metadatas = response['metadatas'][0]
        distances = response['distances'][0]
        # print("RESPONSE")
        # print(response)
        # print("METADATAS")
        # print(metadatas)
        return metadatas, distances
    else:
        return [], []
    # return response'''

# search_chroma_index("find articles, where application of machine learning in medical diagnostics is mentioned")
