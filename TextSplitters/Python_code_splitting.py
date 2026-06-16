from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class NumberChecker:
    def __init__(self, number):
        self.number = number

    def check_even_or_odd(self):
        if self.number % 2 == 0:
            return "The number is even"
        else:
            return "The number is odd"


# Taking input from user
num = int(input("Enter a number: "))

checker = NumberChecker(num)
result = checker.check_even_or_odd()

print(result)
"""


#same thing can be done for other languages too e.g., Language.JAVASCRIPT, Language.JAVA, Language.CPP, Language.RUBY etc.
splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=300, chunk_overlap=0)
chunks = splitter.split_text(text)

print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk}")