from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ```{initial_text}```",
)

template_summarize = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words:\n ```{text}```",
)

llm_en = ChatOpenAI(model="gpt-5-nano", temperature=0)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summarize | llm_en | StrOutputParser()

result = pipeline.invoke(
    {"initial_text": "Langchain é um framework para desenvolver aplicações com LLMs."}
)

print(result)
