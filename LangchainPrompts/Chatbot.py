from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful research assistant.")
]

while True:
    user_input=input("You: ")

    # chat_history.append(user_input)
    chat_history.append(HumanMessage(content=user_input))

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    # response=model.invoke(user_input)
    response=model.invoke(chat_history)

    # chat_history.append(response.content)
    chat_history.append(AIMessage(content=response.content))

    print("Bot: ", response.content)


print ("Final Chat History:")
print (chat_history)