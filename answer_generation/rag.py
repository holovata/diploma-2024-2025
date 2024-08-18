import datetime

from database.db_create import create_tables
from database.db_select import get_all_papers
from database.db_clear import clear_table
from database.db_fetch_and_store import fetch_and_store_papers
from database.db_vectorize import create_chroma_index
from retriever.query_retriever import self_query_search

# from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vectorstore():
    '''# Создание таблиц
    create_tables()

    # Очистка таблицы перед вставкой новых данных
    clear_table("keyword_papers_list")
    print("begin fetching", datetime.datetime.now())
    # Получение и вставка данных из arXiv по ключевому слову
    keyword = "machine learning"  # Измените на нужное ключевое слово
    fetch_and_store_papers(keyword, max_results=200)

    keyword = "computation and language"  # Измените на нужное ключевое слово
    fetch_and_store_papers(keyword, max_results=200)

    keyword = "artificial intelligence"  # Измените на нужное ключевое слово
    fetch_and_store_papers(keyword, max_results=1000)
    print("end fetching", datetime.datetime.now())'''

    import os

    # Получаем путь к текущему файлу
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Путь к директории chroma_store
    chroma_store_dir = os.path.join(current_dir, '..', 'chroma_store')

    # Создание векторного индекса
    print("begin create_chroma_index", datetime.datetime.now())

    # vectorstore, embedding_function = create_chroma_index()
    embedding_function = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        model_name="bert-base-multilingual-cased")
    vectorstore = Chroma(persist_directory=r"C:\Work\mi41\ДИПЛОМ\диплом1\chroma_store", embedding_function=embedding_function)
    print(f"Vector store document count: {len(vectorstore.get()['documents'])}")
    print("end create_chroma_index", datetime.datetime.now())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def process_query(retriever, query):
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs)

    llm = Ollama(model="llama3")

    from langchain.chains import RetrievalQA
    from langchain.chains.llm import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.prompts import PromptTemplate

    prompt = """
            You are an AI helper-assistant that guides scientists through an existing database with arxiv.org articles.
            1. Use ONLY the following pieces of context to answer the question at the end.
            2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
            3. Keep the answer crisp and limited to 2 or 3 sentences for each article.
            4. When mentioning articles, try to provide more information about the articles. Author(s) and the link are a bare minimum.
            5. If you are asked to provide a certain number of articles, provide a list of that exact number of articles from the provided context.


            Context: {context}

            Question: {question}

            Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa(query)["result"]


def main():
    retriever = create_vectorstore()
    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        answer = process_query(retriever, query)
        print(answer)


if __name__ == "__main__":
    main()
