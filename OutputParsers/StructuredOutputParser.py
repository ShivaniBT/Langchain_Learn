from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# these are deprecated in newer versions
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name= "fact_1", description="First fact about the topic"),
    ResponseSchema(name= "fact_2", description="Second fact about the topic"),
    ResponseSchema(name= "fact_3", description="Third fact about the topic")
]

parser= StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about {topic}. \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions" : parser.get_format_instructions()
    }   
)

chain = template | model | parser
parsed_output = chain.invoke({"topic":"Python programming language"})
print("Parsed output:")
print (parsed_output)
print( type(parsed_output))