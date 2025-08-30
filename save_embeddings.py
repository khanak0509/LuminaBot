from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import os 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

os.environ["GOOGLE_API_KEY"] = "AIzaSyAZfeSc6Db1h-0pBxh24XI_8ZIRtSgL3VM"

with open('dataset.txt','r') as f:
    dataset = f.read()
    

document = []
pdf_loader = PyPDFLoader('resume.pdf')
docs = pdf_loader.load()

pdf_text = "\n".join([doc.page_content for doc in docs])

document.append(Document(
    page_content = pdf_text,
    metadata = {"source": "pdf"}
))
document.append(Document(
    page_content = dataset,
    metadata={"source": "audio and image"}
))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(document)


db = FAISS.from_documents(texts,embeddings)

db.save_local('database')
print('saved succressfull')
