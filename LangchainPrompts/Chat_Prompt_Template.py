from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# This will not give expected results

# ChatTemplate =  ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} expert."),
#     HumanMessage(content="Explain in simple temms, what is {topic}?"),
# ])

ChatTemplate =  ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, what is {topic}?"),
])


# Initialize the model
llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

st.header("Topic Explanation UI")

domain_input=st.text_input("Enter the domain of Topic : ")
topic_input=st.text_input("Enter the Topic to be explained : ")

prompt = ChatTemplate.invoke({
    "domain": domain_input,
    "topic": topic_input
})

if st.button("Generate Explanation"):
    response=model.invoke(prompt)
    st.write(response.content)

# print(prompt)