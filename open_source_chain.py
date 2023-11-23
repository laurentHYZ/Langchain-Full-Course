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

# ### Huggingface models
# Langchain is not only about OpenAI´s GPT Models. You can also use free and open source models, for example from huggingface.

# !pip3 install torch==2.0.1
# !pip3 install transformers
# !pip3 install accelerate
# !pip3 install einops
# !pip3 install huggingface_hub
# !pip3 install langchain

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# You need a Huggingface account and API Key from Huggingface and create a token at: https://huggingface.co/settings/tokens.
# You can also set the token in the notebook

# +
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-token"
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"max_length":64, "max_new_tokens":100})

# +
template = """
You are a helpful bot that makes funny jokes about {topic}

Answer: Let's think step by step.
"""

prompt_template = PromptTemplate.from_template(template=template)
chain = LLMChain(llm=llm, prompt=prompt_template)
chain.run("Ducks")
# -

# ### Personal Opinion:
# The output is pretty BAD. The LLM is better when getting some context, which is written in a human like language.

# +
template = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"

Context:
{context}

{query}""".strip()

context = """
The men's high jump event at the 2020 Summer Olympics took place between 30 July and 1 August 2021 at the Olympic Stadium.
33 athletes from 24 nations competed; the total possible number depended on how many nations would use universality places 
to enter athletes in addition to the 32 qualifying through mark or ranking (no universality places were used in 2021).
Italian athlete Gianmarco Tamberi along with Qatari athlete Mutaz Essa Barshim emerged as joint winners of the event following
a tie between both of them as they cleared 2.37m. Both Tamberi and Barshim agreed to share the gold medal in a rare instance
where the athletes of different nations had agreed to share the same medal in the history of Olympics. 
Barshim in particular was heard to ask a competition official "Can we have two golds?" in response to being offered a 
'jump off'. Maksim Nedasekau of Belarus took bronze. The medals were the first ever in the men's high jump for Italy and 
Belarus, the first gold in the men's high jump for Italy and Qatar, and the third consecutive medal in the men's high jump
for Qatar (all by Barshim). Barshim became only the second man to earn three medals in high jump, joining Patrik Sjöberg
of Sweden (1984 to 1992)."""


prompt  = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)
# -

query = "Who won the 2020 Summer Olympics men's high jump?"

chain = LLMChain(llm=llm, prompt=prompt)
chain.run({"query": query, "context": context})
