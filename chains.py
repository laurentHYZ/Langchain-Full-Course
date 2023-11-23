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

# +
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, OpenAI, LLMChain

load_dotenv(find_dotenv())


# +
template = """
Interprete the text and evaluate the text.
sentiment: is the text in a positive, neutral or negative sentiment?
subject: What subject is the text about? Use exactly one word.

Format the output as JSON with the following keys:
sentiment
subject

text: {input}
"""

llm = OpenAI(temperature=0)
prompt_template = PromptTemplate.from_template(template=template)
chain = LLMChain(llm=llm, prompt=prompt_template)
chain.predict(input="I ordered Pizza Salami and it was awesome!")
# -

# ## Sequentials Chains
# Sometimes you want to pass the output from one model to a another model. This can be done with different SequentialsChains

response_template = """
You are a helpful bot that creates a 'thank you' reponse text. 
If customers are unsatisfied, offer them a real world assitant to talk to. 
You will get a sentiment and subject as into and evaluate. 

text: {input}
"""
review_template = PromptTemplate(input_variables=["input"], template=response_template)
review_chain = LLMChain(llm=llm, prompt=review_template)

# +
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[chain, review_chain], verbose=True)

overall_chain.run(input="I ordered Pizza Salami and was aweful!")
# -

# Chains can be more complex and not all sequential chains will be as simple as passing a single string as an argument and getting a single string as output for all steps in the chain

# +
from langchain.chains import SequentialChain

# This is an LLMChain to write a review given a dish name and the experience.
prompt_review = PromptTemplate.from_template(
    template="You ordered {dish_name} and your experience was {experience}. Write a review: "
)
chain_review = LLMChain(llm=llm, prompt=prompt_review, output_key="review")

# This is an LLMChain to write a follow-up comment given the restaurant review.
prompt_comment = PromptTemplate.from_template(
    template="Given the restaurant review: {review}, write a follow-up comment: "
)
chain_comment = LLMChain(llm=llm, prompt=prompt_comment, output_key="comment")

# This is an LLMChain to summarize a review.
prompt_summary = PromptTemplate.from_template(
    template="Summarise the review in one short sentence: \n\n {review}"
)
chain_summary = LLMChain(llm=llm, prompt=prompt_summary, output_key="summary")

# This is an LLMChain to translate a summary into German.
prompt_translation = PromptTemplate.from_template(
    template="Translate the summary to german: \n\n {summary}"
)
chain_translation = LLMChain(
    llm=llm, prompt=prompt_translation, output_key="german_translation"
)

overall_chain = SequentialChain(
    chains=[chain_review, chain_comment, chain_summary, chain_translation],
    input_variables=["dish_name", "experience"],
    output_variables=["review", "comment", "summary", "german_translation"],
)

overall_chain({"dish_name": "Pizza Salami", "experience": "It was awful!"})
# -

# Instead of chaining multiple chains together we can also use an LLM to decide which follow up chain is being used

# +
from langchain.llms import OpenAI
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

positive_template = """You are an AI that focuses on the positive side of things. \
Whenever you analyze a text, you look for the positive aspects and highlight them. \
Here is the text:
{input}"""

neutral_template = """You are an AI that has a neutral perspective. You just provide a balanced analysis of the text, \
not favoring any positive or negative aspects. Here is the text:
{input}"""

negative_template = """You are an AI that is designed to find the negative aspects in a text. \
You analyze a text and show the potential downsides. Here is the text:
{input}"""

# +
prompt_infos = [
    {
        "name": "positive",
        "description": "Good for analyzing positive sentiments",
        "prompt_template": positive_template,
    },
    {
        "name": "neutral",
        "description": "Good for analyzing neutral sentiments",
        "prompt_template": neutral_template,
    },
    {
        "name": "negative",
        "description": "Good for analyzing negative sentiments",
        "prompt_template": negative_template,
    },
]

llm = OpenAI()

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
destination_chains

# +
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["neutral"],
    verbose=True,
)

chain.run("I ordered Pizza Salami for 9.99$ and it was awesome!")
