import dotenv
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    query = "I love cats"

    query_embedding = model.encode(query)
    sentence_embeddings = model.encode(sentences)

    results = util.semantic_search(query_embedding, sentence_embeddings, top_k=2)

    for result in results[0]:
        print(sentences[int(result['corpus_id'])])

if __name__ == "__main__":
    main()