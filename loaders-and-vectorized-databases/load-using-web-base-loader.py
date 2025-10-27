from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://www.langchain.com")
docs = loader.load()

sppliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = sppliter.split_documents(docs)

for chunk in chunks:
    print(chunk)
    print("-" * 30)
