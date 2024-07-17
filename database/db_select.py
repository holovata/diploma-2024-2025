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