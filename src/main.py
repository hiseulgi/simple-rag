import rootutils

ROOT = rootutils.autosetup()

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# scrape data
loader = WebBaseLoader("https://bulbapedia.bulbagarden.net/wiki/Pikachu_(Pok%C3%A9mon)")
data = loader.load()

# split data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
all_splits = text_splitter.split_documents(data)

# convert chunks to vectors
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# ---

# Ollama LLM setup
ollama_llm = "llama3.2"
model_local = ChatOllama(model=ollama_llm, base_url="http://192.168.0.106:11434")

# prompt template
template = """You are an assistant for question-answering tasks.
Use the following documents to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise:

Context:
{context}

Question: 
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# chain setup
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model_local
    | StrOutputParser()
)

# invoke the chain
question = "Can Pikachu use Surf move?"
result = chain.invoke(question)
print("\nRAG Result:")
print(result)

# ---
