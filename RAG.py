from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib


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

        print("Top 3 results:")
        for i, (doc, score) in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Score: {score}")
            print(f"Content: {doc.page_content}")
            print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
            print("------------------------------")

if __name__ == "__main__":
    main()