from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)


documents = [
    "The capital of Germany is Berlin.",
    "TinyLlama is a compact language model optimized for efficient local inference.",
    "The HuggingFacePipeline wrapper allows models to be used locally within LangChain.",
    "Instruction-tuned models often require prompts to be formatted in a specific structure.",
    "Embeddings perform best when each sentence is clear, self-contained, and concise."
]


result=embedding.embed_documents(documents)

print(str(result))