import os
import tempfile
import logging
from backend.utils.base_64_operations import Base64Conversion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class FileUtils:
    def __init__(self, input_data: str, file_name: str, file_extension: str):
        """
        :param input_data: Base64-encoded string
        :param file_name: Desired name for the file (without extension)
        :param file_extension: File extension (e.g., 'pdf', 'jpg')
        """
        self.input_data = input_data
        self.file_name = file_name
        self.file_extension = file_extension

    def file_save(self) -> str | None:
        """
        Validates and decodes base64 input, saves it to a temp file,
        and returns the path. Returns None if validation fails.
        """
        temp_dir = tempfile.mkdtemp()
        filename = f"{self.file_name}.{self.file_extension}"
        file_path = os.path.join(temp_dir, filename)

        try:
            logger.info("Starting base64 validation and decoding.")
            base64_object = Base64Conversion(self.input_data)
            decoded_data = base64_object.is_valid_base64()

            if not decoded_data:
                logger.error("Base64 data is not valid.")
                return None

            with open(file_path, "wb") as f:
                f.write(decoded_data)

            logger.info(f"File successfully saved to temporary path: {file_path}")
            return file_path

        except Exception as e:
            logger.exception("Failed to decode and save base64 file.")
            raise ValueError("An error occurred while processing the base64 data.") from e
