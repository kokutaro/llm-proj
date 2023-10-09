import os
import readline
import datetime
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

OUT_PATH = "out"
CSV_NAME = "retrieval_result"
CSV_HEADER = "Query,Document,Source,Score"
CSV_PREFIX = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)

file_path = os.path.join(OUT_PATH, f"{CSV_NAME}_{CSV_PREFIX}.csv")
f = open(file_path, "w", encoding="utf_8_sig")
f.write(CSV_HEADER + "\n")

while True:
    q = input("Input query: ")
    results = docsearch.similarity_search_with_score("query: " + q)
    lines = []
    for doc, score in results:
        page_content = doc.page_content.replace("passage: ", "")
        print("Socre: ", 1 - score)
        print(page_content)
        print("Source: ", doc.metadata["source"] or "")
        print("-" * 100)
        lines.append(
            f'"{q}","{page_content}","{doc.metadata["source"] or ""}",{1 - score}\n'
        )
    f.writelines(lines)
