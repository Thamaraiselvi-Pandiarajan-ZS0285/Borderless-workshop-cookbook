import os
import logging
from fastapi import UploadFile
from backend.utils.base_64_operations import Base64Conversion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class Base64FileUtils:
    def __init__(self, input_data: str, file_name: str, file_extension: str):
        """
        :param input_data: Base64-encoded string
        :param file_name: Desired name for the file (without extension)
        :param file_extension: File extension (e.g., 'pdf', 'jpg')
        """
        self.input_data = input_data
        self.file_name = file_name
        self.file_extension = file_extension

        # Ensure project-based temp directory
        self.temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(self.temp_dir, exist_ok=True)

    def base64_file_save(self) -> str | None:
        """
        Decodes base64 input and saves it as a file inside the project temp folder.
        Returns the file path or None if failed.
        """
        filename = f"{self.file_name}.{self.file_extension}"
        file_path = os.path.join(self.temp_dir, filename)

        try:
            logger.info("Starting base64 validation and decoding.")
            base64_object = Base64Conversion(self.input_data)
            decoded_data = base64_object.is_valid_base64()

            if not decoded_data:
                logger.error("Base64 data is not valid.")
                return None

            with open(file_path, "wb") as f:
                f.write(decoded_data)

            logger.info(f"File successfully saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.exception("Failed to decode and save base64 file.")
            raise ValueError("An error occurred while processing the base64 data.") from e


class FilePathUtils:
    def __init__(self, file: UploadFile | None, temp_dir: str | None = None):
        # Ensure project-based temp directory
        self.project_temp_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(self.project_temp_dir, exist_ok=True)

        self.file = file
        self.temp_dir = temp_dir or self.project_temp_dir
        logger.info(f"Using temp directory: {self.temp_dir}")

    def file_dir(self) -> str:
        """Returns the temp directory path."""
        return self.temp_dir

    def get_file_name(self) -> str:
        """Get the file name without extension."""
        return os.path.splitext(self.file.filename)[0]

    def get_file_extension(self) -> str:
        """Get the file extension."""
        return os.path.splitext(self.file.filename)[1]

    def file_path_based_file_save(self) -> str:
        """Save file to the temp directory and return full path."""
        file_path = os.path.join(self.temp_dir, self.file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(self.file.file.read())
        logger.info(f"File saved at: {file_path}")
        return file_path
