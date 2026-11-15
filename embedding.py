import dotenv
dotenv.load_dotenv()

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def find_closest(query, sentences):
    query_embedding = model.encode(query)
    sentence_embeddings = model.encode(sentences)

    cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)

    closest_idx = cosine_scores.argmax().item()
    return sentences[closest_idx]

def main():

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    query = "I love cats"

    closest = find_closest(query, sentences)
    print(closest)

if __name__ == "__main__":
    main()