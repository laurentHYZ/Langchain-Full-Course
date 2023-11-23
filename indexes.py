# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# !pip install langchain
# !pip install openai
# !pip install python-dotenv
# !pip install faiss-cpu

# +
from dotenv import load_dotenv
import os

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
API_KEY = os.environ.get("API_KEY")
# -

# ## Loaders  
# To use data with an LLM, documents must first be loaded into a vector database. 
# The first step is to load them into memory via a loader

# +
from langchain.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
docs = loader.load()
# -

# ## Text splitter
# Texts are not loaded 1:1 into the database, but in pieces, so called "chunks". You can define the chunk size and the overlap between the chunks.

# +
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

documents = text_splitter.split_documents(docs)
documents[0]
# -

# ## Embeddings
# Texts are not stored as text in the database, but as vector representations.
# Embeddings are a type of word representation that represents the semantic meaning of words in a vector space.

# +
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
# -

# ## Loading Vectors into VectorDB (FAISS)
# As created by OpenAIEmbeddings vectors can now be stored in the database. The DB can be stored as .pkl file

# +
from langchain.vectorstores.faiss import FAISS
import pickle

vectorstore = FAISS.from_documents(documents, embeddings)

with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
# -

# ## Loading the database
# Before using the database, it must of course be loaded again.

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# ## Prompts
# With an LLM you have the possibility to give it an identity before a conversation or to define how question and answer should look like.

# +
from langchain.prompts import PromptTemplate

prompt_template = """You are a helpful assistant for our restaurant.

{context}

Question: {question}
Answer here:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
# -

# ## Chains
# With chain classes you can easily influence the behavior of the LLM

# +
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}

llm = OpenAI(openai_api_key=API_KEY)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

query = "When does the restaurant open?"
qa.run(query)
# -

# ## Memory
# In the example just shown, each request stands alone. A great strength of an LLM, however, is that it can take the entire chat history into account when responding. For this, however, a chat history must be built up from the different questions and answers. With different memory classes this is very easy in Langchain.

# +
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)
# -

# ## Use Memory in Chains
# The memory class can now easily be used in a chain. This is recognizable, for example, by the fact that when one speaks of "it", the bot understands the rabbit in this context.

# +
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY),
    memory=memory,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": PROMPT},
)


query = "Do you offer vegan food?"
qa({"question": query})
qa({"question": "How much does it cost?"})
