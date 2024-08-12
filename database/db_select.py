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
        "Deep Learning for Economists",
        "A Comparative Study on Automatic Coding of Medical Letters with Explainability",
        "Towards More Trustworthy and Interpretable LLMs for Code through Syntax-Grounded Explanations",
        "Privacy-Preserving Multi-Center Differential Protein Abundance Analysis with FedProt",
        "A Hybrid Approach to Predicting Stock Prices using Machine Learning and Financial Metrics",
        "Using Gradient Boosting Machines for Credit Risk Modeling"
    ]

    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                for title in articles:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1
                            FROM keyword_papers_list
                            WHERE name = %s
                        )
                    """, (title,))

                    exists = cursor.fetchone()[0]
                    if exists:
                        print(f"Article '{title}' exists in the database.")
                    else:
                        print(f"Article '{title}' does not exist in the database.")
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while checking articles:", error)


# Call the function to check articles
check_articles_exist()