from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    # template = "Give me the name, age and city of a fictional person. \n {format_instructions}",
    template = "Give me 5 facts about {topic}. \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions" : parser.get_format_instructions() 
    }   
)

# prompt = template.format()

# print ("Prompt sent to the model:")
# print (prompt)


# result = model.invoke(prompt)
# print("Raw output from the model:")
# print (result)

# parsed_output = parser.parse(result.content)


#--------------USING THE JSON OUTPUT PARSER IN A CHAIN----------------
chain = template | model | parser
parsed_output = chain.invoke({"topic":"Albert Einstein"})
print("Parsed output:")
print (parsed_output)
print( type(parsed_output))

# print(f"\n Name: {parsed_output['name']}")

print ("\n Fact1 about Albert Einstein:", parsed_output['facts'][0])


#-------------the flaw is , we cannot define schema for the output we want from the model using JsonOutputParser directly. 
# For that we need PydanticOutputParser or (StructuredOutputParser - this is depricated now). see that file----------------