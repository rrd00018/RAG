from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from typing import List
import argparse
import os
import shutil


DATA_PATH ="data"
CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_data()
    chunks = split_text(documents)
    save_to_database(chunks)

def load_data():
    loader = DirectoryLoader(DATA_PATH,glob="*.pdf")
    docs = loader.load()
    print("Se han cargado " + str(len(docs)) + " documentos")
    return docs

def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    a = chunks[10]
    print(a.page_content)
    print(a.metadata)

    return chunks

def save_to_database(chunks: List[Document]):
    database = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    chunks_with_ids = calculate_chunks_ids(chunks)

    existing_items = database.get(include=[])
    existing_items_id = set(existing_items["ids"]) #Cuando accedo a la base de datos esta entre otras cosas devuelve los id de las tuplas seleccionadas. Por tanto para acceder a ellos se usa el campo ids siempre

    print(f"Numero de archivos en la base de datos: {len(existing_items)}")

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_items_id:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Añadiendo {len(new_chunks)} chunks en la base de datos")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        database.add_documents(new_chunks, ids=new_chunks_ids)
    else:
        print("No hay nuevos documentos que añadir")

    print("Acceso a base de datos terminado")


def calculate_chunks_ids(chunks: List[Document]):
    last_page_id = None
    current_index = 0

    for chunk in chunks:
        source_file = chunk.metadata.get("source")
        current_page = chunk.metadata.get("page")
        current_page_id = f"{source_file}:{current_page}"

        if(current_page_id == last_page_id):
            current_index += 1
        else:
            current_index = 0

        chunk_id = f"{current_page_id}:{current_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
