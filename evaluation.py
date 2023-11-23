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
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
API_KEY = os.environ.get("API_KEY")

# +
from langchain.prompts.prompt import PromptTemplate

llm = OpenAI(temperature=0)

_PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
What grade do you give from 0 to 10, where 0 is the lowest (very low similarity) and 10 is the highest (very high similarity)?
"""

PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)
# -

context_examples = [
    {
        "question": "Do you offer vegetarian or vegan options?",
        "context": "Yes, we have a range of dishes to cater to vegetarians and vegans",
    },
    {
        "question": "What are the hours of operation for your restaurant?",
        "context": "Our restaurant is open from 11 a.m. to 10 p.m. from Monday to Saturday. On Sundays, we open at 12 p.m. and close at 9 p.m.",
    },
]
QA_PROMPT = "Answer the question based on the  context\nContext:{context}\nQuestion:{question}\nAnswer:"
template = PromptTemplate(input_variables=["context", "question"], template=QA_PROMPT)
qa_chain = LLMChain(llm=llm, prompt=template)
predictions = qa_chain.apply(context_examples)
predictions

# +
from langchain.evaluation.qa import ContextQAEvalChain

eval_chain = ContextQAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(
    context_examples, predictions, question_key="question", prediction_key="text"
)
graded_outputs
