from pydantic import BaseModel

class EncodeFileResponse(BaseModel):
    response: str
    statusCode: int
    message: str


def build_encode_file_response(base64_string: str, status_code: int, message: str) -> EncodeFileResponse:
    return EncodeFileResponse(
        response=base64_string,
        statusCode=status_code,
        message=message
    )
