from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
import os
from CONFIG import config
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def debug_log(message):
    print(f"[DEBUG] {message}")


def process_query(query, persist_dir="C:\Work\mi41\ДИПЛОМ\диплом1\chroma_store"):
    try:
        debug_log(f"Persist directory: {persist_dir}")
        collection_name = 'papers_collection'

        debug_log("Initializing Chroma vector store")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=config.embedding_function,
                          collection_name=collection_name)

        debug_log("Similarity search")
        result = vectordb.similarity_search(query, k=2)

        '''debug_log("Creating retriever")
        retriever = vectordb.as_retriever()

        debug_log(f"Retrieving documents for query: {query}")
        retrieved_docs = retriever.invoke(query)
        debug_log(f"Retrieved documents: {retrieved_docs}")

        debug_log("Initializing LLM")
        llm = Ollama(model="llama3")'''

        prompt = """
            You are an AI helper-assistant that guides scientists through an existing database with arxiv.org articles.
            1. Use ONLY the following pieces of context to answer the question at the end.
            2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.
            3. Keep the answer crisp and limited to 2 or 3 sentences for each article.
            4. When mentioning articles, try to provide more information about the articles. Author(s) and the link are a bare minimum.
            5. If you are asked to provide a certain number of articles, provide a list of that exact number of articles from the provided context.

            Context: {context}

            Question: {question}

            Helpful Answer:"""

        return result

    except RuntimeError as e:
        debug_log(f"RuntimeError: {e}")
        debug_log("There was an issue with the vector store. Please check the persist directory and files.")
        return "An error occurred while processing the query."


def main():
    while True:
        query = input("\nQuery: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        debug_log(f"Processing new query: {query}")
        answer = process_query(query)
        print(answer)


if __name__ == "__main__":
    main()
