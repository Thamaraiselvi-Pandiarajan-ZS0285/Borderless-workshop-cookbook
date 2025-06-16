import base64
from abc import ABC, abstractmethod

class Base64Conversion(ABC):
    def __init__(self, input_data: str):
        """
        You may pass either a temporary file path or a base64 string.
        Both can be used depending on the context (encode or decode).
        """
        self.input_data = input_data

    @abstractmethod
    def do_base64_encoding(self):
        pass

    @abstractmethod
    def do_base64_decoding(self):
        pass

    def is_valid_base64(self):
        try:
            if isinstance(self.input_data, str):
                data = self.input_data.encode('utf-8')
            decoded = base64.b64decode(data, validate=True)
            encoded = base64.b64encode(decoded)
            return encoded == data.strip()
        except Exception:
            return False