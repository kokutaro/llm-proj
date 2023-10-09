import os
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    WikipediaLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

text_splitter_chunk_size = int(os.environ.get("TEXT_SPLITTER_CHUNK_SIZE", 300))
text_splitter_chunk_overlap = int(os.environ.get("TEXT_SPLITTER_CHUNK_OVERLAP", 150))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=text_splitter_chunk_size,
    chunk_overlap=text_splitter_chunk_overlap,
)


def load_pdf(path: str) -> List[Document]:
    documents = PyPDFLoader(path).load()
    for doc in documents:
        doc.metadata["source"] = {
            "fileName": path.split("/")[-1],
            "page": doc.metadata["page"],
        }

    documents = text_splitter.split_documents(documents)
    for doc in documents:
        doc.page_content = "passage: " + doc.page_content

    return documents


def load_docx(path: str) -> List[Document]:
    documents = UnstructuredWordDocumentLoader(path).load()
    for doc in documents:
        print(doc.metadata)
        doc.metadata["source"] = {
            "fileName": path.split("/")[-1],
            "page": doc.metadata["page"],
        }

    documents = text_splitter.split_documents(documents)
    for doc in documents:
        doc.page_content = "passage: " + doc.page_content

    return documents


def load_wiki(
    search_query: str, lang="ja", doc_content_chars_max=10000
) -> List[Document]:
    documents = WikipediaLoader(
        search_query, lang, doc_content_chars_max=doc_content_chars_max
    ).load()
    for doc in documents:
        doc.metadata["source"] = doc.metadata["title"]

    documents = text_splitter.split_documents(documents)
    for doc in documents:
        doc.page_content = (
            "passage: " + doc.metadata["title"] + " > " + doc.page_content
        )
    return documents
