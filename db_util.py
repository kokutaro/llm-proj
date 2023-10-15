import os
import dotenv
from langchain.vectorstores import PGVector

dotenv.load_dotenv()


def get_connection_string():
    return PGVector.connection_string_from_db_params(
        driver=os.environ.get("POSTGRES_DB_DRIVER", "psycopg2"),
        host=os.environ.get("POSTGRES_DB_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_DB_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "postgres"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )
