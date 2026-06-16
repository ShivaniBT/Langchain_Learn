from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

# llm=HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation",
# )

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
# result = model.invoke("What is the capital of France?")
result = model.invoke("Tell me a joke about computers.")
print(result.content)
