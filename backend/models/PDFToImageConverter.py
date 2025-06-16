from abc import ABC, abstractmethod

class PDFToImageConverter(ABC):
    @abstractmethod
    def convert(self, pdf_path, output_folder, dpi=300):
        pass