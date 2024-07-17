# main.py
import datetime

from database.db_create import create_tables
from database.db_select import get_all_papers
from database.db_clear import clear_table
from database.db_fetch_and_store import fetch_and_store_papers
from database.db_vectorize import create_chroma_index, search_chroma_index
from retriever.query_retriever import self_query_search

def main():
    # Создание таблиц
    create_tables()

    # Очистка таблицы перед вставкой новых данных
    clear_table("keyword_papers_list")

    # Получение и вставка данных из arXiv по ключевому слову
    keyword = "machine learning"  # Измените на нужное ключевое слово
    print("begin fetching", datetime.datetime.now())
    fetch_and_store_papers(keyword, max_results=70)
    print("end fetching", datetime.datetime.now())
    # Создание векторного индекса
    print("begin create_chroma_index", datetime.datetime.now())
    client, collection, papers = create_chroma_index()
    print("end create_chroma_index", datetime.datetime.now())
    print(collection.count())
    # Выполнение поиска по векторному индексу
    # query = "find articles, where application of machine learning in medical diagnostics is mentioned"  # Пример поискового запроса
    query = "Найди статьи за 2023 год по теме машинного обучения"
    top_k = 10
    print("begin search_chroma_index", datetime.datetime.now())
    results, distances = search_chroma_index(query, top_k)
    print("end search_chroma_index", datetime.datetime.now())

    filtered_results = self_query_search(query, collection)

    print("Результаты фильтрации:")
    # Печать результатов поиска
    print(f"Top {top_k} results for query '{query}':")
    for i, (result, dist) in enumerate(zip(results, distances)):
        print(f"{i + 1}. {result['name']} (Distance: {dist:.4f})")
        print(f"Authors: {result['authors']}")
        print(f"URL: {result['url']}")
        print(f"Abstract: {result['abstract']}")
        print(f"Keyword: {result['keyword']}")
        print(f"Categories: {result['categories']}")
        print(f"Year: {result['year']}")
        print(f"Eprint: {result['eprint']}")
        print()

if __name__ == "__main__":
    main()
