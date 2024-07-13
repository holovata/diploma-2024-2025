from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma

llm = Ollama(model="llama3")


def self_query_search(query, collection):
    llm = Ollama(model_name="llama3")  # Подключение к модели Ollama LLaMA 7B
    retriever = SelfQueryRetriever.from_llm(llm, vectorstore=collection)

    # Выполнение запроса
    results = retriever.retrieve(query)
    return results
