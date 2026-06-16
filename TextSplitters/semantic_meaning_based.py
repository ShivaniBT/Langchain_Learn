from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

text = """
Cricket is one of the most popular sports in the world, especially in countries like India, England, and Australia. 
The game requires teamwork, strategy, and physical endurance. 
Different formats such as Test cricket, One Day Internationals, and T20 matches offer unique experiences to players and fans.

Artificial intelligence is rapidly transforming modern industries. 
Machine learning, a subset of AI, enables systems to learn from data and improve over time. 
Applications of AI include healthcare, finance, autonomous vehicles, and natural language processing.

Climate change is a major global challenge affecting ecosystems and human life. 
Rising temperatures are causing glaciers to melt and sea levels to rise. 
Reducing carbon emissions and adopting renewable energy sources are critical steps toward sustainability.
"""

chunker = SemanticChunker(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5
)

documents = chunker.create_documents([text])
print(f"Number of chunks created: {len(documents)}")
for i, doc in enumerate(documents):
    print(f"\nChunk {i+1}:\n{doc.page_content}")