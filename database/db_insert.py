# database/db_insert.py

import psycopg2
from .db_config import get_db_connection


def insert_paper(name, authors, url, abstract, keyword, subjects, year, eprint):
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute('''
                    INSERT INTO keyword_papers_list (name, authors, url, abstract, keyword, subjects, year, eprint) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (name, authors, url, abstract, keyword, subjects, year, eprint))

                connection.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while inserting data:", error)
