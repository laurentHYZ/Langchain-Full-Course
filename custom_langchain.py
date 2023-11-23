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

# ### Custom Chains
#
# Sometimes you might not have a predefined solution for your problem, you might want more control or just 
# get a better understanding of what chains actually do.

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# +
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain


chain = LLMChain(
    prompt=PromptTemplate.from_template("tell us a joke about {topic}"),
    llm=ChatOpenAI(),
)

chain.run("chains", callbacks=[StdOutCallbackHandler()])

# +
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.schema import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate


class WikipediaArticleChain(Chain):
    """
    Custom chain for generating a brief Wikipedia article on a given topic.
    """

    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    output_key: str = "article"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        prompt_value = self.prompt.format_prompt(**inputs)
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        if run_manager:
            run_manager.on_text("Generated Wikipedia article on given topic")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Does not support async")

    @property
    def _chain_type(self) -> str:
        return "wikipedia_article_chain"


from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

wikipedia_prompt = PromptTemplate.from_template("Write a brief Wikipedia-style article on the topic {topic}")

chain = WikipediaArticleChain(
    prompt=wikipedia_prompt,
    llm=ChatOpenAI(),
)


# +
# Running our custom chain with a given topic
result = chain.run("quantum physics", callbacks=[StdOutCallbackHandler()])

print(type(result))

# -

output_dict = chain({"topic": "Quantum Physics"}, callbacks=[StdOutCallbackHandler()])
print(type(output_dict))

output_dict


