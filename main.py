from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyA83KimfLvcMNWd7P8PEP7yzNC9V6ZnUDM"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    timeout=None,
    max_retries=2,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local("database", embeddings, allow_dangerous_deserialization=True)

query = "Who introduced ChatGPT in 2022?"
results = db.similarity_search(query, k=3)  

final_texts = []
for i, res in enumerate(results, 1):
    final_texts.append(res.page_content)

combined_text = " ".join(final_texts)

prompt = PromptTemplate(
    template="""
give and of query - {query} \n from {combined_text} 

""",
input_variables=['query','combined_text']
)

chain = prompt | llm 
print(chain.invoke({
    'query' : query,
    'combined_text' : combined_text
}).content)
