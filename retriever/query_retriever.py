#query_retriever.py

from langchain_community.llms import Ollama
from langchain.llms import BaseLLM
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

def self_query_search(query, collection):
    metadata_field_info = [
        AttributeInfo(
            name="name",
            description="Title of the scientific paper as listed on the arxiv.org website.",
            type="string",
        ),
        AttributeInfo(
            name="authors",
            description="List of authors of the scientific paper, separated by commas. Each author is usually listed with their full name.",
            type="string",
        ),
        AttributeInfo(
            name="url",
            description="Direct URL to the scientific paper on the arxiv.org website, where the full text and additional details can be accessed.",
            type="string",
        ),
        AttributeInfo(
            name="abstract",
            description="Summary of the scientific paper, providing a brief overview of the research, methodology, and findings.",
            type="string",
        ),
        AttributeInfo(
            name="keyword",
            description="Main keywords associated with the scientific paper, indicating the primary topics and areas of research.",
            type="string",
        ),
        AttributeInfo(
            name="categories",
            description="Categories assigned to the scientific paper, indicating the specific fields or disciplines it belongs to (e.g., Computer Science, Physics).",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="Year in which the scientific paper was published on arxiv.org.",
            type="string",
        ),
        AttributeInfo(
            name="eprint",
            description="E-print identifier of the scientific paper on arxiv.org, which is a unique alphanumeric string used to locate the paper.",
            type="string",
        ),
    ]

    llm = Ollama(model="llama3")
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=collection,
        document_contents="Abstract of the scientific paper",
        metadata_field_info=metadata_field_info,
        verbose=True
    )

    # Выполнение запроса
    results = retriever.retrieve(query)
    return results

