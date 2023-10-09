from typing import List
import os
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "intfloat/multilingual-e5-large"

model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs=model_kwargs)

connection_string = PGVector.connection_string_from_db_params(
    driver=os.environ.get("POSTGRES_DB_DRIVER", "psycopg2"),
    host=os.environ.get("POSTGRES_DB_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_DB_PORT", "5432")),
    database=os.environ.get("POSTGRES_DB", "postgres"),
    user=os.environ.get("POSTGRES_USER", "postgres"),
    password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
)

docsearch = PGVector(connection_string, embeddings, "embedding_store")


def store_embbeding(input: List[Document]) -> List[str]:
    return docsearch.add_documents(input)


def main():
    while True:
        q = input("Input passage to store: ")
        result = docsearch.add_texts(["passage: " + q])
        print(result)


if __name__ == "__main__":
    main()
