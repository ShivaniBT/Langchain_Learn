from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

prompt1= PromptTemplate(
    template="Suggest a joke about {topic}",
    input_variables=["topic"]
)

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

chain = RunnableSequence(prompt1, model,parser)

parallel_chain = RunnableParallel({
    "Joke": RunnablePassthrough(),
    # "WordCount": RunnableLambda(lambda inputs: len(inputs.split()))
    "WordCount": runnable_word_counter
})

final_chain = chain | parallel_chain

result = final_chain.invoke({"topic":"cats"})
print("Joke and Word Count:")
print(result)

final_result = "{} \nThe joke has {} words.".format(result["Joke"], result["WordCount"])
print("\nFinal Result:")
print(final_result)