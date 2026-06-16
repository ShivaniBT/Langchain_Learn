from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


prompt1= PromptTemplate(
    template="Suggest a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the given joke more interesting: {joke}",
    input_variables=["joke"]
)

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

chain = RunnableSequence(prompt1, model,parser,prompt2,model, parser)

result = chain.invoke({"topic":"cats"})

print("Joke:")
print(result)