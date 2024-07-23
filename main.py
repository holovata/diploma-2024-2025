import datetime

from database.db_create import create_tables
from database.db_select import get_all_papers
from database.db_clear import clear_table
from database.db_fetch_and_store import fetch_and_store_papers
from database.db_vectorize import create_chroma_index
from retriever.query_retriever import self_query_search

from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def main():
    # Создание таблиц
    create_tables()

    # Очистка таблицы перед вставкой новых данных
    clear_table("keyword_papers_list")

    # Получение и вставка данных из arXiv по ключевому слову
    keyword = "machine learning"  # Измените на нужное ключевое слово
    print("begin fetching", datetime.datetime.now())
    fetch_and_store_papers(keyword, max_results=200)
    print("end fetching", datetime.datetime.now())

    '''    if results is None or distances is None:
            print("Error during query execution.")
            return'''

    '''    print("Результаты фильтрации:")
        # Печать результатов поиска
        # print(f"Top {top_k} results for query '{query}':")
        # for i, (result, dist) in enumerate(zip(results, distances)):
        for i, result in response:
            print(f"{i + 1}. {result['name']} (Distance: {dist:.4f})")
            print(f"Authors: {result['authors']}")
            print(f"URL: {result['url']}")
            print(f"Abstract: {result['abstract']}")
            print(f"Keyword: {result['keyword']}")
            print(f"Categories: {result['categories']}")
            print(f"Year: {result['year']}")
            print(f"Eprint: {result['eprint']}")
            print()'''

    # Создание векторного индекса
    print("begin create_chroma_index", datetime.datetime.now())
    vectorstore = create_chroma_index()
    print(f"Vector store document count: {len(vectorstore.get()['documents'])}")
    print("end create_chroma_index", datetime.datetime.now())
    retriever = vectorstore.as_retriever()

    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        retrieved_docs = retriever.invoke(query)
        print(retrieved_docs)

'''       # Prompt
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible.
        {context}
        Question: {question}
        Helpful Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        # Local LLM
        ollama_llm = "llama3"
        model_local = ChatOllama(model=ollama_llm)

        # Chain
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model_local
                | StrOutputParser()
        )'''


'''
# Выполнение поиска по векторному индексу
query = "find articles about Schatten-p Low Rank Approximation"
    top_k = 5
    print("begin self_query_search", datetime.datetime.now())
    # results, distances = self_query_search(query, vectorstore)
    response = vectorstore.similarity_search(query=query, k=top_k)
    metadatas = [doc.metadata for doc in response]
    print("end self_query_search", datetime.datetime.now())'''


'''    print("Результаты фильтрации:")
    # Печать результатов поиска
    for i, metadata in enumerate(metadatas):
        print(f"{i + 1}. {metadata['name']}")
        print(f"Authors: {metadata['authors']}")
        print(f"URL: {metadata['url']}")
        print(f"Abstract: {metadata['abstract']}")
        print(f"Keyword: {metadata['keyword']}")
        print(f"Categories: {metadata['categories']}")
        print(f"Year: {metadata['year']}")
        print(f"Eprint: {metadata['eprint']}")
        print()'''



if __name__ == "__main__":
    main()
