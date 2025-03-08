#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from langchain.vectorstores import FAISS
from tqdm import tqdm
from config.appConfigProvider import AppConfigProvider

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

dataProvider = AppConfigProvider()

# Load environment variables
gpt_enabled = dataProvider.get_by_key_value("GPT_ENABLED")
persist_directory = dataProvider.get_by_key_value('PERSIST_DIRECTORY')
source_directory = dataProvider.get_by_key_value('SOURCE_DIRECTORY')
embeddings_model_name = dataProvider.get_by_key_value('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    if not documents:
        print("No new documents to load")
        return ""
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


# def does_vectorstore_exist(persist_directory: str) -> bool:
#     """
#     Checks if vectorstore exists
#     """
#     if os.path.exists(os.path.join(persist_directory, 'index')):
#         if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
#                 os.path.join(persist_directory, 'chroma-embeddings.parquet')):
#             list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
#             list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
#             # At least 3 documents are needed in a working vectorstore
#             if len(list_index_files) > 3:
#                 return True
#     return False


def data_Ingest_main():
    if gpt_enabled == "true":
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        # if does_vectorstore_exist(persist_directory):
        #     # Delete existing vectorstore
        #     shutil.rmtree(persist_directory)
        #     print(f"Deleted existing vectorstore at {persist_directory}")

        # Create and store a new vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        if texts != "":
            print(f"Creating embeddings. May take some minutes...")
            db = FAISS.from_documents(texts,embeddings)
            # db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory,
            #                            client_settings=CHROMA_SETTINGS)
            db.save_local(persist_directory)
            db = None

        print(f"Ingestion complete! You can now run emilyGPT to query your documents \n")
        print(f"-----------------------------------------End of ingestion--------------------------------\n")
        return {"status": True}
    else:
        print(f"GPT is not enabled check system settings")
        return {"status": False}


if __name__ == "__main__":
    data_Ingest_main()
