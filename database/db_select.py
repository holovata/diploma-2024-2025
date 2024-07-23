# database/db_select.

import psycopg2
from database.db_config import get_db_connection


def get_all_papers():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT * FROM keyword_papers_list')
                records = cursor.fetchall()
                return records

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching data:", error)
        return []


# print(get_all_papers())

def check_articles_exist():
    articles = [
        ("Quantifying Systemic Risk through Network Analysis", 2022),
        ("Machine Learning in Finance: A Survey", 2020),
        ("Deep Learning for Credit Risk Assessment", 2019),
        ("A Study on Machine Learning-Based Predictive Models for Stock Market Forecasting", 2020),
        ("Neural Networks in Finance: An Overview and Some Applications", 2018),
        ("Unsupervised Learning for Anomaly Detection in Financial Time Series", 2022)
    ]

    connection = get_db_connection()
    cursor = connection.cursor()

    for title, year in articles:
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM keyword_papers_list
                WHERE name = %s AND year = %s
            )
        """, (title, year))

        exists = cursor.fetchone()[0]
        if exists:
            print(f"Article '{title}' ({year}) exists in the database.")
        else:
            print(f"Article '{title}' ({year}) does not exist in the database.")

    cursor.close()
    connection.close()


# Call the function to check articles
# check_articles_exist()