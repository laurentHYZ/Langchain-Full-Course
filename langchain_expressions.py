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

from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("tell me a joke about {input}")

chain = LLMChain(llm=llm, prompt=prompt)
chain.predict(input="A pig")

# ### First pipe

chain = prompt | llm


chain.invoke({"input": "A pig"})

from langchain.schema.output_parser import StrOutputParser

chain = prompt | llm | StrOutputParser()
chain.invoke({"input": "A pig"})

prompt = ChatPromptTemplate.from_template(
    "tell me 5 jokes about {input}"
    )

chain = prompt | llm.bind(stop=["\n"]) | StrOutputParser()

chain.invoke({"input": "pigs"})

# ### OPENAI Functions

functions = [
    {
      "name": "joke",
      "description": "A joke",
      "parameters": {
        "type": "object",
        "properties": {
          "setup": {
            "type": "string",
            "description": "The setup for the joke"
          },
          "punchline": {
            "type": "string",
            "description": "The punchline for the joke"
          }
        },
        "required": ["setup", "punchline"]
      }
    }
  ]

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
chain = (
    prompt 
    | llm.bind(function_call={"name": "joke"}, functions= functions) 
    | JsonOutputFunctionsParser()
)

chain.invoke(input={"input": "bears"})


# ### Working with vectorstores

# +
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough


vectorstore = Chroma.from_texts(["Cats are typically 9.1 kg in weight.", 
                                 "Cats have retractable claws.", 
                                 "A group of cats is called a clowder.", 
                                 "Cats can rotate their ears 180 degrees.", 
                                 "The world's oldest cat lived to be 38 years old."], 
                                embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# -

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

chain.invoke("how old is the oldest cat?")

from operator import itemgetter

question = {"bla": "test", "x": "wuff"}
itemgetter("bla")


get_bla = itemgetter("bla")
get_bla(question)

# +
from langchain.schema.runnable import RunnablePassthrough

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

# chain = {
#     "context": itemgetter("question") | retriever,
#     "question": itemgetter("question"),
#     "language": itemgetter("language")
# } | prompt | model | StrOutputParser()

chain = {
    "context": (lambda x: x["question"]) | retriever, 
    "question": (lambda x: x["question"]), 
    "language": (lambda x: x["language"])
} | prompt | llm | StrOutputParser()

# -

chain.invoke({"question": "how old is the oldest cat?", "language": "german"})

# !pip install duckduckgo-search

from langchain.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

# +
template = """turn the following user input into a search query for a search engine:

{input}"""

prompt = ChatPromptTemplate.from_template(template)
# -

chain = prompt | llm | StrOutputParser() | search

chain.invoke({"input": "whats the name of the oldest cat?"})

# ### Arbitrary functions

# +
from langchain.schema.runnable import RunnableLambda

def length_function(text):
    return len(text)

def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)

def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])

prompt = ChatPromptTemplate.from_template("what is {a} + {b}")

chain = {
    "a": itemgetter("foo") | RunnableLambda(length_function),
    "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")} | RunnableLambda(multiple_length_function)
} | prompt | llm | StrOutputParser()
# -

chain.invoke({"foo": "bar", "bar": "gah"})

# ### Interface

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

for s in chain.stream({"topic": "bears"}):
    print(s.content, end="")

chain.invoke({"topic": "bears"})

chain.batch([{"topic": "bears"}, {"topic": "cats"}])

async for s in chain.astream({"topic": "bears"}):
    print(s.content, end="")
await chain.ainvoke({"topic": "bears"})
await chain.abatch([{"topic": "bears"}])
