from pydantic import BaseModel, Field

class Documents(BaseModel):
    documents: list[str] = Field(default=[], strict=True)
    filenames: list[str] = Field(default=[], strict=True)

class Query(BaseModel):
    query: str = Field(default="", strict=True)
    n: int = Field(default=10, strict=True)

class QueryDocument(BaseModel):
    query: str = Field(default="", strict=True)
    docs: list[str] = Field(default=[], strict=True)