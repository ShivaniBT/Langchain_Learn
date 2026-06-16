from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


prompt1= PromptTemplate(
    template="Suggest a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the given joke: {topic}",
    input_variables=["topic"]
)

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

chain = RunnableSequence(prompt1, model,parser)

parallel_chain = RunnableParallel({
    "Joke": RunnablePassthrough(),
    "ExplainedJoke": RunnableSequence(prompt2, model, parser)
})

final_chain = chain | parallel_chain

result = final_chain.invoke({"topic":"cats"})
print("Joke and Explained Joke:")
print(result)