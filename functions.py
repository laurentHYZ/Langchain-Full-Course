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

# ## Function calling
#
# This new feature proves beneficial in a wide array of situations. It can assist in designing chatbots capable of interacting with various APIs, facilitate task automation, and enable extraction of organized information from inputs expressed in natural language.

# !pip install --upgrade langchain
# !pip install python-dotenv
# !pip install openai

# Ensure you got the AT LEAST Version 0.0.200

# +
import pkg_resources


def print_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        print(f"The version of the {package_name} library is {version}.")
    except pkg_resources.DistributionNotFound:
        print(f"The {package_name} library is not installed.")


print_version("langchain")

# +
from dotenv import load_dotenv
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import json

load_dotenv()



# -

# First, you need to define a function. This one mimics a database query with a "fake" price.

def get_pizza_info(pizza_name: str):
    pizza_info = {
        "name": pizza_name,
        "price": "10.99",
    }
    return json.dumps(pizza_info)


functions = [
    {
        "name": "get_pizza_info",
        "description": "Get name and price of a pizza of the restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "pizza_name": {
                    "type": "string",
                    "description": "The name of the pizza, e.g. Salami",
                },
            },
            "required": ["pizza_name"],
        },
    }
]


def chat(query):
    response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
    messages=[{"role": "user", "content": query}],
    functions=functions)
    message = response["choices"][0]["message"]
    return message


chat("What is the capital of france?")

query = "How much does pizza salami cost?"
message = chat(query)
message

# +
if message.get("function_call"):
    function_name = message["function_call"]["name"]
    pizza_name = json.loads(message["function_call"]["arguments"]).get("pizza_name")
    print(pizza_name)
    function_response = get_pizza_info(
        pizza_name=pizza_name
    )

    second_response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": query},
        message,
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
        },
    ])

second_response
# -

# ### LangChain 
#
# LangChain allows you to use functions too, but currently (15.06.2023) it is kind of hacky to do it. 
#

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, ChatMessage

llm = ChatOpenAI(model="gpt-3.5-turbo-0613")
message = llm.predict_messages(
    [HumanMessage(content="What is the capital of france?")], functions=functions
)
message

query = "How much does Pizza Salami cost in the restaurant?"
message_pizza = llm.predict_messages([HumanMessage(content=query)], functions=functions)
message_pizza

message.additional_kwargs

message_pizza.additional_kwargs

# +
import json

pizza_name = json.loads(message_pizza.additional_kwargs["function_call"]["arguments"]).get("pizza_name")
pizza_name
# -

pizza_api_response = get_pizza_info(pizza_name=pizza_name)
pizza_api_response

# +


second_response = llm.predict_messages(
    [
        HumanMessage(content=query),
        AIMessage(content=str(message_pizza.additional_kwargs)),
        ChatMessage(
            role="function",
            additional_kwargs={
                "name": message_pizza.additional_kwargs["function_call"]["name"]
            },
            content=pizza_api_response
        ),
    ],
    functions=functions,
)
second_response
# -

# ### Use Tools
#
# You can convert Tools to functions, both the Tools provided by langchain and also your own tools. The Tool you see is there to show you the interface of a tool, it actually does nothing really do anything useful ;-) 

# +
from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


class StupidJokeTool(BaseTool):
    name = "StupidJokeTool"
    description = "Tool to explain jokes about chickens"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return "It is funny, because AI..."

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("joke tool does not support async")


# +
from langchain.tools import format_tool_to_openai_function, MoveFileTool


tools = [StupidJokeTool(), MoveFileTool()]
functions = [format_tool_to_openai_function(t) for t in tools]
functions
# -

query = "Why does the chicken cross the road? To get to the other side"
output = llm.predict_messages([HumanMessage(content=query)], functions=functions)
output

question = json.loads(output.additional_kwargs["function_call"]["arguments"]).get("__arg1")
tool_response = tools[0].run(question)
tool_response

second_response = llm.predict_messages(
    [
        HumanMessage(content=query),
        AIMessage(content=str(output.additional_kwargs)),
        ChatMessage(
            role="function",
            additional_kwargs={
                "name": output.additional_kwargs["function_call"]["name"]
            },
            content="""
                {tool_response}
            """,
        ),
    ],
    functions=functions,
)
second_response

# ### OpenAI Functions Agent

# +
from langchain import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]
# -

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

agent.run("What is the capital of france?")

agent.run("What is 100 devided by 25?")
