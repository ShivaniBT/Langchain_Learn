from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader,WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


url = "https://en.wikipedia.org/wiki/Science" #can use list of urls too
loader = WebBaseLoader(url)

docs = loader.load()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)   
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Answer the following question :\n {question}\n based on given text - {text}",
    input_variables=["question","text"]
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"question":"What is the history of the word Science?","text": docs[0].page_content})

print("Answer:")
print(result)
# print(f"Number of documents loaded: {len(docs)}")
# print (docs[0].page_content[:500])  # Print the first 500 characters of the first document
# print (docs[0].metadata)  # Print metadata of the first document