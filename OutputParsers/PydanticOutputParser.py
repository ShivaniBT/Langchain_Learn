from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Student(BaseModel):
    name: str = Field(description="The name of the student")
    age: int = Field(gt=18, description="The age of the student")
    city: str = Field(description="The city where the student lives")

parser = PydanticOutputParser(pydantic_object=Student)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {ethnicity} student. \n {format_instructions}",
    input_variables=["ethnicity"],
    partial_variables={
        "format_instructions" : parser.get_format_instructions()
    }   
)

prompt = template.invoke({"ethnicity":"Indian"})
result = model.invoke(prompt)
parsed_output = parser.parse(result.content)

print ("Prompt sent to the model:")
print (prompt)
print ("\n")

# chain = template | model | parser
# parsed_output = chain.invoke({"ethnicity":"Indian"})


print("Parsed output:")
print (parsed_output)
print( type(parsed_output))