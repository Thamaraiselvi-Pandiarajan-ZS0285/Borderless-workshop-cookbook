import base64
from abc import ABC, abstractmethod

class Base64Conversion(ABC):
    def __init__(self, input_data: str):
        """
        You may pass either a temporary file path or a base64 string.
        This can be used depending on the context (encode or decode).
        """
        self.input_data = input_data

    @abstractmethod
    def do_base64_encoding(self) -> str:
        """
        Subclasses should implement this to perform Base64 encoding.
        """
        pass

    @abstractmethod
    def do_base64_decoding(self) -> bytes:
        """
        Subclasses should implement this to perform Base64 decoding.
        """
        pass


class Base64Utils:
    @staticmethod
    def is_valid_base64(s: str) -> bool:
        try:
            if not isinstance(s, str):
                return False
            data = s.encode("utf-8")
            decoded = base64.b64decode(data, validate=True)
            encoded = base64.b64encode(decoded)
            return encoded == data.strip()
        except Exception:
            return False