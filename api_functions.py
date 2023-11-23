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

# ### Function Calling and API Calls
#
# A lot of projects with Langchain are just scripts executed on a local computer. In the "real world", production ready
# services run in some kind of standardized wrapper, like an API inside a Docker container. This is an approach how you might
# interact with a REST-API with an LLM Chain (which might later also be used inside an API withh a single POST endpoint)

# +
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# +
import requests

def get_todos(completed=None):
    params = {'completed': completed} if completed is not None else None
    response = requests.get('https://fastapilangchain-1-w0858112.deta.app/todos', params=params)
    return response.json()

def create_todo(todo):
    response = requests.post('https://fastapilangchain-1-w0858112.deta.app/todos', json=todo)
    return response.json()

def update_todo(id, todo):
    response = requests.put(f'https://fastapilangchain-1-w0858112.deta.app/todos/{id}', json=todo)
    return response.json()

def delete_todo(id):
    response = requests.delete(f'https://fastapilangchain-1-w0858112.deta.app/todos/{id}')
    return response.status_code  



# -

api_functions = {
    "get_todos": get_todos,
    "create_todo": create_todo,
    "update_todo": update_todo,
    "delete_todo": delete_todo
}

functions = [
    {
        "name": "get_todos",
        "description": "Get a list of todos, optionally filtered by their completion status",
        "parameters": {
            "type": "object",
            "properties": {
                "completed": {
                    "type": "boolean",
                    "description": "Whether to only return completed todos",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_todo",
        "description": "Create a new todo",
        "parameters": {
            "type": "object",
            "properties": {
                "todo": {
                    "type": "object",
                    "description": "The new todo to be created",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "The id of the todo",
                        },
                        "task": {
                            "type": "string",
                            "description": "The task of the todo",
                        },
                        "is_completed": {
                            "type": "boolean",
                            "description": "Whether the task is completed",
                            "default": False
                        },
                    },
                    "required": ["task"],
                },
            },
            "required": ["todo"],
        },
    },
    {
        "name": "update_todo",
        "description": "Update an existing todo",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The id of the todo to update",
                },
                "todo": {
                    "type": "object",
                    "description": "The updated todo",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The updated task of the todo",
                        },
                        "is_completed": {
                            "type": "boolean",
                            "description": "The updated completion status of the task",
                        },
                    },
                    "required": ["task"],
                },
            },
            "required": ["id", "todo"],
        },
    },
    {
        "name": "delete_todo",
        "description": "Delete an existing todo",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "integer",
                    "description": "The id of the todo to delete",
                },
            },
            "required": ["id"],
        },
    }
]


# +
query="I want to walk my dog in the afternoon"

response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": query}],
        functions=functions,
    )
message = response["choices"][0]["message"]

message

# +
import json

if message.get("function_call"):
    function_name = message["function_call"]["name"]
    function_args_json = message["function_call"].get("arguments", {})
    function_args = json.loads(function_args_json)

    api_function = api_functions.get(function_name)

    if api_function:
        result = str(api_function(**function_args)) 
    else:
        print(f"Function {api_function} not found")

result

# +
second_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "user", "content": query},
        message,
        {
            "role": "function",
            "name": function_name,
            "content": result,
        },
    ],
)

second_response
