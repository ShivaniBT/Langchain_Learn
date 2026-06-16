from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough, RunnableLambda,RunnableBranch
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

prompt1= PromptTemplate(
    template="Give an article on the topic - {topic}",
    input_variables=["topic"]
)

prompt2= PromptTemplate(
    template="Summarize the following article in a concise manner: {article}",
    input_variables=["article"]
)

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

chain = RunnableSequence(prompt1, model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = chain | branch_chain
result = final_chain.invoke({"topic":"The impact of artificial intelligence on modern healthcare systems"})
print("Article and Summary (if applicable):")
print(result)