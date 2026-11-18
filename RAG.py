from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib
import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://api.gapgpt.app/v1"
            )

def llm(query, context):

    system_prompt = f"""You are a medical information assistant. Answer the user's question using ONLY
    the provided source documents below.

      Rules you must follow:
      1. Base your answer ONLY on the provided sources — do not use outside knowledge
      2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer
      this question."
      3. Cite sources using [Source N] notation whenever you use information from them
      4. If sources contradict each other, mention the contradiction
      5. Be concise — answer in 3-5 sentences unless the question requires more detail
      6. Never speculate or make assumptions beyond what the sources say

      Sources:
      {context}"""

    response = client.chat.completions.create(
        model="gpt-5.6-terra",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()


model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    collection_name="medical_data",
    embedding_function=model,
    persist_directory='./chroma',
    collection_metadata={'hnsw:space': 'cosine'}
)

def read_pdf_file(file_name):
    reader = PdfReader(file_name)
    docs = [Document(page_content=page.extract_text(), metadata={"source": file_name, "page": i+1}) for i, page in enumerate(reader.pages)]
    
    return docs

def generate_id(docs):
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]

def main():
    docs = read_pdf_file("symptoms.pdf")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    split_chunks = text_splitter.split_documents(docs)
    ids = generate_id(split_chunks)

    vectorstore.add_documents(split_chunks, ids=ids)

    while True:
        user_input = input("Enter your query (or 'q' to quit): ")
        if user_input.lower() == 'q':   
            break

        results = vectorstore.similarity_search_with_score(user_input, k=3)

        parts = []

        for i, (doc, score) in enumerate(results):
            parts.append(f"[Source {i+1}] | Page {doc.metadata.get('page')} | cosine similarity {score: .2f}: {doc.page_content}")

        context = "\n\n_____\n\n".join(parts)
        answer = llm(user_input, context)

        print(answer)

        # print("Top 3 results:")
        # for i, (doc, score) in enumerate(results):
        #     print(f"Result {i+1}:")
        #     print(f"Score: {score}")
        #     print(f"Content: {doc.page_content}")
        #     print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
        #     print("------------------------------")

if __name__ == "__main__":
    main()