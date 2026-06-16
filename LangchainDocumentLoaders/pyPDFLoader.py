from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

loader = PyPDFLoader("Sample.pdf")

docs = loader.load()

print(f"Number of pages loaded: {len(docs)}")

print(docs[0].page_content[:500])  # Print the first 500 characters of the first page
print(docs[0].metadata)  # Print metadata of the first page