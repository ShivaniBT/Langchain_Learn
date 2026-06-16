from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# docs = loader.load()
docs = loader.lazy_load()  # Use lazy loading for large directories

# print(f"Number of documents loaded: {len(docs)}")

# print (docs[1].page_content[:500])  # Print the first 500 characters of the first document
# print (docs[1].metadata)  # Print metadata of the first document

for doc in docs:
    print(doc.metadata)  # Print metadata of each document