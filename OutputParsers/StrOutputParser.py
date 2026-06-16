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


# first prompt - detailed report of the topic
template1= PromptTemplate(
    template="Write a detailed report about {topic}.",
    input_variables=["topic"]
)

# second prompt - 4-5 line summary of the report
template2= PromptTemplate(
    template="Write a 4-5 line summary on the following text. /n {text}.",
    input_variables=["text"]
)

# prompt1 = template1.invoke({"topic":"Black hole"})
# result1 = model.invoke(prompt1) 

# prompt2 = template2.invoke({"text":result1.content})
# result2 = model.invoke(prompt2)

# print("4-5 line summary of the report on Black hole:")
# print(result2.content)



# --------------NOW WE WILL USE STR OUTPUT PARSER TO GET THE SUMMARY----------------

parser= StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black hole"})

print ("4-5 line summary of the report on Black hole:")
print (result)