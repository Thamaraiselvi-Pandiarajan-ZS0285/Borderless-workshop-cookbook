import tempfile
from backend.utils.base_64_operations import Base64Conversion


class FileUtils:
    def __init__(self, input_data: str, file_name:str, file_extension:str):
        self.input_data = input_data
        self.file_name = file_name
        self.file_extension = file_extension

    def file_save(self):
        try:
            base64_object = Base64Conversion(self.input_data)
            decoded_data = base64_object.is_valid_base64()
        except Exception as e:
            raise ValueError("Invalid base64 data") from e

        suffix = f".{self.file_extension}" if not self.file_extension.startswith('.') else self.file_extension

        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.write(decoded_data)
        temp_file_path = temp_file.name
        temp_file.close()

        print(f"Temp file saved at: {temp_file_path}")
        return temp_file_path
