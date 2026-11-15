import dotenv
from openai import embeddings
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]

def main():
    embeddings = model.encode(sentences)
s
    similarities = model.similarity(embeddings, embeddings)
    print(similarities.shape)

if __name__ == "__main__":
    main()