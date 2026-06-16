from langchain_community.document_loaders import PyPDFLoader, CSVLoader

loader = CSVLoader(file_path="sample_data.csv", encoding="utf-8")

docs = loader.load()

print(f"Number of rows loaded: {len(docs)}")