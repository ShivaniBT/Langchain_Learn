from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Sample.pdf")

docs = loader.load()


# text = """
# here are several code snippets demonstrating the use of Langchain for various tasks, including loading documents from directories and web pages, processing text with language models, and counting words using runnable lambdas.
# 1. The first snippet shows how to load CSV files using the CSVLoader from Langchain Community Document Loaders. It loads a CSV file and prints the number of rows loaded.   
# """

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator='')
# chunks = splitter.split_text(text)

chunks = splitter.split_documents(docs)

print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk}")