from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Suggest the catchy blog title about topic- {topic}?",
    input_variables=["topic"]
)

topic = input("Enter the topic for blog title suggestion: ")

# -------lLMCahin way(deprecated)---------
# chain = LLMChain(
#     prompt=prompt,
#     llm=model
# )


parser= StrOutputParser()

task_chain = prompt | model | parser
result = task_chain.invoke({"topic":topic})

print("Suggested Blog Title:")
print(result)
