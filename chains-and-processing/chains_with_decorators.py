from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()


@chain
def square(x_dict: dict) -> dict:
    x = x_dict["x"]
    return {"square_result": x * x}


question_template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!",
)

square_question_template = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}",
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = question_template | model
chain_square = square | square_question_template | model

result = chain.invoke({"name": "Victor"})
print(result.content)

result_square = chain_square.invoke({"x": 10})
print(result_square.content)