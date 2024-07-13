# database/db_fetch_and_store.py

import arxiv
from .db_insert import insert_paper

def fetch_and_store_papers(keyword, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for result in client.results(search):
        name = result.title
        authors = ', '.join([author.name for author in result.authors])
        url = result.entry_id
        abstract = result.summary
        categories = ', '.join(result.categories)
        year = result.updated.year
        eprint = result.entry_id

        insert_paper(name, authors, url, abstract, keyword, categories, year, eprint)
