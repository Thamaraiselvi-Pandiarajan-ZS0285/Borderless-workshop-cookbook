from pydantic import BaseModel

class PaperItemizerResponse(BaseModel):
    response: list
    statusCode: int
    message: str


def build_paper_itemizer_response(response: list, status_code: int, message: str) -> PaperItemizerResponse:
    return PaperItemizerResponse(
        response=response,
        statusCode=status_code,
        message=message
    )
