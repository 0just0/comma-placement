from pydantic import BaseModel


class InputText(BaseModel):
    input_text: str


class FixedText(BaseModel):
    text_with_commas: str
    original_text: str
