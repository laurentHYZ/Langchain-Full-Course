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

# !pip install langchain

# +
import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"]
# -

llm = OpenAI(model_name="text-ada-001")
llm("Tell me a joke")

# ## PromptTemplates 
# Prompt Templates makes creating a prompt for different usecases easier than using f-strings and interpolate the videos

template = """
Interprete the text and evaluate the text.
sentiment: is the text in a positive, neutral or negative sentiment?
subject: What subject is the text about? Use exactly one word.

Format the output as JSON with the following keys:
sentiment
subject

text: {input}
"""

# +
from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(template=template)
chain = LLMChain(llm=llm, prompt=prompt_template)
chain.predict(input="I ordered Pizza Salami and it was awesome!")
# -

# ## Real World example with ResponseSchema, Templates, Chains and OutputParsers
# There were two issues with the output: The output also contains text and the output is just a string which can not just the converted to a dictionary. Lets make it better with a more complex example

# +
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

sentiment_schema = ResponseSchema(
    name="sentiment",
    description="Is the text positive, neutral or negative? Only provide these words",
)
subject_schema = ResponseSchema(
    name="subject", description="What subject is the text about? Use exactly one word."
)
price_schema = ResponseSchema(
    name="price",
    description="How expensive was the product? Use None if no price was provided in the text",
)

response_schemas = [sentiment_schema, subject_schema, price_schema]
# -

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
format_instructions

# +
template = """
Interprete the text and evaluate the text.
sentiment: is the text in a positive, neutral or negative sentiment?
subject: What subject is the text about? Use exactly one word.

Just return the JSON, do not add ANYTHING, NO INTERPRETATION!

text: {input}

{format_instructions}

"""

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_template(template=template)

format_instructions = parser.get_format_instructions()

messages = prompt.format_messages(
    input="I ordered Pizza Salami for 9.99$ and it was awesome!",
    format_instructions=format_instructions,
)

chat = ChatOpenAI(temperature=0.0)
response = chat(messages)
response
# -

output_dict = parser.parse(response.content)
output_dict


