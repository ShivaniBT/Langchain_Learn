from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,SystemMessage, AIMessage


# chat template
chat_template= ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{customer_issue}")
])

# load chat history

chat_history = []

with open ("chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

print("Chat History Loaded:")
print(chat_history)


# create prompt

prompt= chat_template.invoke({
    "chat_history": chat_history,
    "customer_issue": "Where is my refund?"
})

print("Generated Prompt:")
print(prompt)

