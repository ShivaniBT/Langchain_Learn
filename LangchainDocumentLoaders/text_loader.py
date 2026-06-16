from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

loader = TextLoader("Cricket.txt", encoding="utf8")

docs=loader.load()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Summarize the following text:\n\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"text": docs[0].page_content})

print("Summary:")
print(result)

# print(docs)

# print(docs[0].page_content)

# print (type(docs))
# print (type(docs[0]))