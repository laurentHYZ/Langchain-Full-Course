# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: autogpt
#     language: python
#     name: python3
# ---

# ## Why Langchain
# Langchain allows you to interact with your models in a standardized way and lets your easily compose components from langchain to fulfill your special usecase

# !pip install python-dotenv
# !pip install openai

# +
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())



# -

def chat(input):
    messages = [{"role": "user", "content": input}]
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=messages,
    temperature=0)
    return response.choices[0].message["content"]


output = chat("What is the capital of france")
output

# +
question = "What is the capital of france"

prompt = """
Be very funny when answering questions
Question: {question}
""".format(
    question=question
)

print(prompt)
chat(prompt)
