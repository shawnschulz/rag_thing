from typing import List, Optional

from pydantic import BaseModel, Field

class Document(BaseModel):
    """Information from a document."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    title: Optional[str] = Field(default=None, description="The title of the article or piece of documentation")
    body: Optional[str] = Field(
        default=None, description="The content of the article itself"
    )
    source_url: Optional[str] = Field(
        default=None, description="The source url of the documentation"
    )
    outgoing_links: Optional[str] = Field(
        default=None, description="A list of any relevant urls found in the document"
    )
    date_information: Optional[str] = Field(
        default=None, description="Any information about the date the article was published"
    )

class Data(BaseModel):
    """Extracted data about documentation."""

    # Creates a model so that we can extract multiple entities.
    documentation: List[Document]
