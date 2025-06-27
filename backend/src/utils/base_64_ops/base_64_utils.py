import base64
import binascii
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def encode_file_to_base64(file_path: Union[str, Path]) -> str:
    """
    Reads a file and encodes its content to a base64 string.
    """
    file_path = Path(file_path)
    try:
        logger.debug(f"Reading file for encoding: {file_path}")
        with file_path.open('rb') as file:
            file_data = file.read()
        encoded_data = base64.b64encode(file_data).decode('utf-8')
        logger.debug("File successfully encoded to base64.")
        return encoded_data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to encode file: {e}")
        raise

def decode_base64(base64_data: str) -> bytes:
    """
    Decodes a base64 string and returns the raw binary data.
    Raises ValueError if the input is not valid base64.
    """
    try:
        logger.debug("Decoding base64 string to bytes.")
        decoded_data = base64.b64decode(base64_data, validate=True)
        logger.debug("Base64 decoding successful.")
        return decoded_data
    except (binascii.Error, ValueError):
        logger.warning("Invalid base64 input.")
        raise ValueError("Invalid base64 input")
    except Exception as e:
        logger.error(f"Unexpected error during base64 decoding: {e}")
        raise


def is_valid_base64(s: str) -> bool:
    """
    Validates if the provided string is a valid base64-encoded string.
    """
    try:
        base64.b64decode(s, validate=True)
        logger.debug("Valid base64 string.")
        return True
    except (binascii.Error, ValueError):
        logger.debug("Invalid base64 string.")
        return False
    except Exception as e:
        logger.error(f"Error during base64 validation: {e}")
        return False
