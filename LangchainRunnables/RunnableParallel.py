from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


prompt1= PromptTemplate(
    template="Generate linkedIn post about this given topic about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate twitter post about this given topic about {topic}",
    input_variables=["topic"]
)

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "LinkedInPost": RunnableSequence(prompt1, model,parser),
        "TwitterPost": RunnableSequence(prompt2, model,parser)
    }
)

parallel_chain_result = parallel_chain.invoke({"topic":"Artificial Intelligence"})

print("LinkedIn Post:")
print(parallel_chain_result["LinkedInPost"])    
print("\nTwitter Post:")
print(parallel_chain_result["TwitterPost"])