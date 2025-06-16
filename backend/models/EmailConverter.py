from abc import ABC, abstractmethod

class EmailConverter(ABC):
    @abstractmethod
    def convert_to_pdf(self, email_data, output_path):
        pass