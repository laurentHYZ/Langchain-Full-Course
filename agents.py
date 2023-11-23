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

# # Prepration
# Loading env variables and vectorDB

# +
from dotenv import load_dotenv
import os
import pickle

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
API_KEY = os.environ.get("API_KEY")
# -

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# # Agents
#
# Agents use an LLM to determine which actions to perform and in what order. An action can be either using a tool and observing its output or returning it to the user. To use an agent, in addition to the concept of an LLM, it is important to understand a new concept and that of a "tool".

# ## Tools
#
# Tools are functions that agents can use to interact with the world. These tools can be common utilities (e.g. search), other chains, or even other agents.

# +
from langchain.agents import load_tools
from langchain.llms import OpenAI

llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY)

tool_names = ["llm-math"]
tools = load_tools(tool_names, llm=llm)
tools
# -

from langchain.agents import Tool
tool_list = [
    Tool(
        name = "Math Tool",
        func=tools[0].run,
        description="Tool to calculate, nothing else"
    )
]

# +
from langchain.agents import initialize_agent

agent = initialize_agent(tool_list, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)
agent.run("How are you?")
# -

agent.run("What is 100 devided by 25?")

# # Custom Tools
#
# You can also create your own tools by creating a class that inherits from BaseTool.

# +
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class CustomSearchTool(BaseTool):
    name = "restaurant search"
    description = "useful for when you need to answer questions about our restaurant"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        store = vectorstore.as_retriever()
        docs = store.get_relevant_documents(query)
        text_list = [doc.page_content for doc in docs]
        return "\n".join(text_list)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


# +
from langchain.agents import AgentType

tools = [CustomSearchTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# -

agent.run("When does the restaurant open?") 
