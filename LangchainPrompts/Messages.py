from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful research assistant."),
    HumanMessage(content="Tell me about Langchain."),
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print (messages)