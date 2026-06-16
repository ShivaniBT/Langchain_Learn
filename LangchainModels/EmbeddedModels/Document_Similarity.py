from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query ="Tell me about Sachin."

document_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores=cosine_similarity([query_embedding], document_embeddings)
print (str(similarity_scores))
most_similar_index = np.argmax(similarity_scores)

print("Query:", query)
print("Most similar document:", documents[most_similar_index])
print("Similarity score:", similarity_scores[0][most_similar_index])