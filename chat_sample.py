import os
import chainlit as cl
import hashlib
import psycopg2
import db_util
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from uuid import uuid4
from chainlit.input_widget import Select
import doc_loader

MODEL_NAME = "intfloat/multilingual-e5-large"

max_size_mb = int(os.environ.get("MAX_SIZE_MB", 10))
max_files = int(os.environ.get("MAX_FILES", 4))

model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs=model_kwargs)
docsearch = PGVector(db_util.get_connection_string(), embeddings, "embedding_store")


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            )
        ]
    ).send()

    value = settings["Model"]
    file_ask = cl.AskFileMessage(
        content=f"Please upload up to {max_files} `.pdf` or `.docx` files to begin.",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ],
        max_size_mb=max_size_mb,
        max_files=max_files,
        timeout=86400,
        raise_on_timeout=False,
    )
    # await file_ask.send()
    await cl.Text(content=value, name="text1", display="side").send()
    await cl.Text(content="Page", name="text2", display="page").send()
    content = "Here is image1, a nice image of a cat! As well as text1 and text2!"

    await cl.Message(
        content=content,
    ).send()


@cl.on_message
async def main(message: str):
    res = await docsearch.asimilarity_search_with_relevance_scores(message)
    elements = []
    file_names = []
    for doc, score in res:
        file_path = os.path.join(
            "uploads", doc.metadata["doc_uuid"], doc.metadata["source"]
        )
        elements.append(
            cl.Pdf(name=doc.metadata["source"], display="side", path=file_path)
        )
        file_names.append(doc.metadata["source"])
    # Do any post processing here
    # Send the response
    await cl.Message(content=",".join(set(file_names)), elements=elements).send()


@cl.on_file_upload(
    accept=[
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ],
    max_files=max_files,
    max_size_mb=max_size_mb,
)
async def on_file_upload(files: any):
    if len(files) == 1:
        content = f"Processing `{files[0]['name']}`..."
    else:
        files_names = [f"`{f['name']}`" for f in files]
        content = f"Processing {', '.join(files_names)}..."
    msg = cl.Message(content=content, author="Chatbot")
    await msg.send()

    con = psycopg2.connect(
        host=os.environ.get("POSTGRES_DB_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_DB_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "postgres"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )
    cur = con.cursor()

    sql = """
INSERT INTO uploaded_documents
(uuid, file_name, sha512, content_type)
VALUES (%s, %s, %s, %s)
"""

    dup_chk_sql = """
SELECT COUNT(1) AS COUNT FROM uploaded_documents
WHERE sha512 = %s
"""

    content = """
## Upload result

|File name|Result|Chunk size|
|:--------|:-----|---------:|
"""

    docs = []
    for file in files:
        file_name = file["name"]
        file_content = file["content"]
        content_type = file_name.split(".")[-1]
        content += f"|{file_name}|"
        sha512 = hashlib.sha512(file_content).hexdigest()
        cur.execute(dup_chk_sql, [sha512])
        (count,) = cur.fetchone()
        if int(count) > 0:
            content += "Skipped|-|\n"
            continue
        uuid = str(uuid4())
        save_dir = os.path.join("uploads", uuid)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = os.path.join(save_dir, file_name)
        f = open(save_dir, "wb")
        cur.execute(sql, (uuid, file_name, sha512, content_type))
        f.write(file_content)
        docs.extend(doc_loader.load_documents(save_dir, uuid))
        content += f"Saved|{len(docs)}|\n"

    await docsearch.aadd_documents(docs)

    await cl.Message(content=content, author="Chatbot").send()

    con.commit()
    cur.close()
    con.close()
