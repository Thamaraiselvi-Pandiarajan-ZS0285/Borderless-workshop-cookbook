from pdf2image import convert_from_path
import os
from abc import ABC, abstractmethod
from backend.utils.file_utils import FilePathUtils

class PDFToImageConverter(ABC):
    @abstractmethod
    def convert(self, pdf_path, output_folder, dpi=300):
        pass

class PdfToImageConverter(PDFToImageConverter):
    def convert(self, pdf_path, output_folder, dpi=500, source_type="unknown"):
        """
        Convert PDF to images and track source type.

        :param pdf_path: Path to input PDF
        :param output_folder: Folder to save images
        :param dpi: DPI for conversion quality
        :param source_type: 'email_attachment' or 'direct_mail_pdf'
        """
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)

        file_utils = FilePathUtils(file=None, temp_dir=None)
        output_folder = file_utils.file_dir()

        images = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []

        for i, image in enumerate(images):
            img_path = os.path.join(output_folder, f"page_{i+1}.png")
            image.save(img_path, "PNG")
            image_paths.append(img_path)

        print(f"Converted {len(images)} pages from {source_type} at {dpi} DPI.")
        return image_paths