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

# !pip install doctran
# !pip install langchain

from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document
import json

complaint_email_content = """
Dear Mustang Motors Team,

I'm writing to express my dissatisfaction with a recent car purchase (Mustang GT). Regrettably, the car hasn't lived up to my expectations, and I would like to request a refund.

The car's functionality and performance have not matched the quality advertised through your marketing initiatives and previous customer testimonials. The innovative design and features, as touted by your R&D department, seem inconsistent with the actual vehicle.

Could you please guide me through the refund process? I trust in your ability to resolve this issue promptly and satisfactorily.

Thank you for your attention to this matter.

Best regards,

Max Müller
"""

# +
from langchain.document_transformers import DoctranPropertyExtractor

documents = [Document(page_content=complaint_email_content)]
properties = [
    {
        "name": "category",
        "description": "The type of email this is.",
        "type": "string",
        "enum": ["complaint", "refund_request", "product_feedback", "customer_service", "other"],
        "required": True,
    },
    {
        "name": "mentioned_product",
        "description": "The product mentioned in this email.",
        "type": "string",
        "required": True,
    },
    {
        "name": "issue_description",
        "description": "A brief explanation of the problem encountered with the product.",
        "type": "string",
        "required": True,
    }
]
property_extractor = DoctranPropertyExtractor(properties=properties, openai_api_model="gpt-3.5-turbo")

# -

extracted_document = await property_extractor.atransform_documents(
    documents, properties=properties
)
print(json.dumps(extracted_document[0].metadata, indent=2))

from langchain.schema import Document
from langchain.document_transformers import DoctranQATransformer


documents = [Document(page_content=complaint_email_content)]
qa_transformer = DoctranQATransformer(openai_api_model="gpt-3.5-turbo")
transformed_document = await qa_transformer.atransform_documents(documents)

print(json.dumps(transformed_document[0].metadata, indent=2))

from langchain.document_transformers import DoctranTextTranslator

documents = [Document(page_content=complaint_email_content)]
qa_translator = DoctranTextTranslator(language="german", openai_api_model="gpt-3.5-turbo")

translated_document = await qa_translator.atransform_documents(documents)
translated_document[0].page_content
