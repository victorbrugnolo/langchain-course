from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("gpt-5.pdf")
docs = loader.load()

sppliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = sppliter.split_documents(docs)

print(f"Total chunks: {len(chunks)}")
