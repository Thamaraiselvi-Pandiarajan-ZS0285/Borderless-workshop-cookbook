import base64
import os
import logging

from backend.utils.base_64_operations import Base64Conversion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FileToBase64(Base64Conversion):
    def do_base64_encoding(self) -> str:
        return self.do_base64_encoding_by_file_path()

    def do_base64_encoding_by_file_path(self) -> str:
        try:
            if not os.path.isfile(self.input_data):
                raise FileNotFoundError(f"File not found: {self.input_data}")

            with open(self.input_data, "rb") as file:
                encoded_data = base64.b64encode(file.read()).decode("utf-8")
                logger.info(f"File encoded successfully: {self.input_data}")
                return encoded_data

        except Exception as e:
            logger.error(f"Error encoding file to base64: {e}", exc_info=True)
            raise RuntimeError("Failed to encode file to base64.") from e

    def do_base64_decoding(self) -> bytes:
        # You can decode the base64 string back into bytes here
        try:
            return base64.b64decode(self.input_data, validate=True)
        except base64.binascii.Error as e:
            logger.error("Invalid Base64 input.", exc_info=True)
            raise ValueError("Invalid base64 string provided.") from e

    @staticmethod
    def decode_base64_to_file(base64_string: str, output_file_path: str) -> None:
        try:
            decoded_data = base64.b64decode(base64_string, validate=True)

            with open(output_file_path, "wb") as f:
                f.write(decoded_data)

            logger.info(f"Base64 decoded and saved to: {output_file_path}")

        except base64.binascii.Error as e:
            logger.error("Invalid Base64 input.", exc_info=True)
            raise ValueError("Invalid base64 string provided.") from e

        except Exception as e:
            logger.error(f"Error decoding base64 to file: {e}", exc_info=True)
            raise RuntimeError("Failed to decode base64 string to file.") from e
