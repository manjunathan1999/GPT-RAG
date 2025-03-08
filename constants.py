import os
from chromadb.config import Settings

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/db"

CHROMA_SETTINGS = Settings(
        chroma_db_impl = 'duckdb+parquet',
        persist_directory = PERSIST_DIRECTORY,
        anonymized_telemetry = False
)

embeddings_model_name = ""
persist_directory = ''
model_type = ""
model_path = ""
model_n_ctx = ""
model_n_batch = ""
target_source_chunks = ""