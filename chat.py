import os
import io
import torch
import chainlit as cl
from uuid import uuid4
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from doc_loader import load_pdf, load_docx

MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
TEMP_DIRECTORY = "temp"

max_size_mb = int(os.environ.get("MAX_SIZE_MB", 10))
max_files = int(os.environ.get("MAX_FILES", 4))
text_splitter_chunk_size = int(os.environ.get("TEXT_SPLITTER_CHUNK_SIZE", 300))
text_splitter_chunk_overlap = int(os.environ.get("TEXT_SPLITTER_CHUNK_OVERLAP", 150))

if not os.path.exists(TEMP_DIRECTORY):
    os.mkdir(TEMP_DIRECTORY)

template = """ユーザーの質問に答えるために、以下の文脈の一部を使用して回答を生成してください。
答えがわからない場合は、ただわからないと言い、答えを作ろうとしないでください。
回答には必ず "SOURCES" の部分を返してください。"PAGE"の部分があれば、それも返してください。
"SOURCES"の部分は、あなたが答えを得た文書の出典を示すものでなければなりません。
回答例

---

○○○○
SOURCES: xyz
PAGE: 23
---

開始
----------------
{summaries}
"""

template = template.replace("\n", "<NL>")

connection_string = PGVector.connection_string_from_db_params(
    driver=os.environ.get("POSTGRES_DB_DRIVER", "psycopg2"),
    host=os.environ.get("POSTGRES_DB_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_DB_PORT", "5432")),
    database=os.environ.get("POSTGRES_DB", "postgres"),
    user=os.environ.get("POSTGRES_USER", "postgres"),
    password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
)

model_kwargs = {"device": "cuda"}

messages = [
    SystemMessagePromptTemplate.from_template(template),
    HumanMessagePromptTemplate.from_template("質問: {question}<NL>システム:"),
]
prompt = ChatPromptTemplate.from_messages(messages)
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME, device_map="auto", torch_dtype=torch.bfloat16
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    temperature=0.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
)
llm = HuggingFacePipeline(pipeline=pipe)
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs=model_kwargs)
docsearch = PGVector(connection_string, embeddings, "embedding_store")
chain_type_kwargs = {"prompt": prompt, "verbose": True}
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot", url="https://cdn-icons-png.flaticon.com/512/8649/8649595.png"
    ).send()
    await cl.Avatar(
        name="Error", url="https://cdn-icons-png.flaticon.com/512/8649/8649595.png"
    ).send()
    await cl.Avatar(
        name="User",
        url="https://media.architecturaldigest.com/photos/5f241de2c850b2a36b415024/master/w_1600%2Cc_limit/Luke-logo.png",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=f"Please upload up to {max_files} `.pdf` or `.docx` files to begin.",
            accept=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ],
            max_size_mb=max_size_mb,
            max_files=max_files,
            timeout=86400,
            raise_on_timeout=False,
        ).send()
    content = ""
    if len(files) == 1:
        content = f"Processing `{files[0].name}`..."
    else:
        files_names = [f"`{f.name}`" for f in files]
        content = f"Processing {', '.join(files_names)}..."
    msg = cl.Message(content=content, author="Chatbot")
    await msg.send()
    all_texts = []
    # Process each file uploaded by the user
    for file in files:
        # Create an in-memory buffer from the file content
        bytes = io.BytesIO(file.content)

        # Get file extension
        extension = file.name.split(".")[-1]

        # Initialize the text variable
        texts = []
        temp_path = os.path.join("temp", str(uuid4()))
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)

        file_path = os.path.join(temp_path, file.name)
        f = open(file_path, "wb")
        f.write(bytes.read())
        f.close()
        # Read the file
        if extension == "pdf":
            texts = load_pdf(file_path)
        elif extension == "docx":
            texts = load_docx(file_path)

        os.unlink(file_path)
        os.rmdir(temp_path)

        # Add the chunks and metadata to the list
        all_texts.extend(texts)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(all_texts))]

    await docsearch.aadd_documents(all_texts)
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", all_texts)
    if len(files) == 1:
        content = f"`{files[0].name}` processed. You can now ask questions!"
    else:
        files_names = [f"`{f.name}`" for f in files]
        content = f"{', '.join(files_names)} processed. You can now ask questions."
    msg.content = content
    msg.author = "Chatbot"
    await msg.update()
    cl.user_session.set("llm_chain", qa)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here
    # Send the response
    await cl.Message(content=res["answer"]).send()
