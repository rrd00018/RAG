from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


import argparse

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text :str):
    embedding_function = get_embedding_function()
    database = Chroma(persist_directory=CHROMA_PATH,embedding_function=embedding_function)

    results = database.similarity_search_with_score(query_text, k=5)
    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) #De los resultados de la busqueda por similaridad, los ordena por la puntuacion y los concatena en un unico string
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return response

if __name__ == "__main__":
    main()