#!/usr/bin/env python3
import time
import logging
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.llms.llamacpp import LlamaCpp
from constants import embeddings_model_name, persist_directory, model_type, model_path, model_n_ctx, model_n_batch, target_source_chunks
from contextlib import contextmanager



def create_llama_model():
    mute_stream = False
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
    if model_type == "LlamaCpp":
        return LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    elif model_type == "GPT4All":
        return GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    else:
        raise ValueError(f"Model type {model_type} not supported!")


def emilyGPTmain(query: str):
    template = """
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
        <ctx>
        {context}
        </ctx>
        <hs>
        {history}
        </hs>
        {question}
        Answer:
        """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    try:
        llm = create_llama_model()
        # mute_stream = False
        hide_source = False
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = FAISS.load_local(persist_directory,embeddings)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        qa = retrieval_qa(llm=llm, chain_type="stuff", verbose=False, retriever=retriever,
                                    return_source_documents=not hide_source, chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    })
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [
        ] if hide_source else res['source_documents']
        end = time.time()

        output = {
            "question": query,
            "response": answer,
            "execution_time": round(end - start, 2)
        }
        source_documents = []
        for document in docs:
            source_documents.append({
                "reference": document.metadata["source"],
                "page_content": document.page_content
            })
            output["source_documents"] = source_documents
            return output
        else:
            logging.info(f"GPT is not enabled check system settings")
            return {
                "question": query,
                "response": "",
                "execution_time": 0,
                "source_documents": []
            }
    except Exception as err:
        print(f"Unexpected emilyIngest {err=}, {type(err)=}")
        logging.exception(f"Unexpected {err=}, {type(err)=}")
        raise err


if __name__ == "__main__":
    emilyGPTmain()
