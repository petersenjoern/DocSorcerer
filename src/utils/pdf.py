# Utils working with PDF (text and non-text (image) based)

import os
import pathlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import fitz
import pypdf

from llama_index.readers.base import BaseReader
from llama_index.schema import Document


def convert_pdf_in_images(pdf_dir: Path, work_dir: str) -> int:
    """
    The convert_pdf_in_images function converts a PDF file into images.

    :param pdf_dir: Path: Specify the path of the pdf file to be converted
    :param work_dir: str: Specify the directory where the images will be saved
    :return: The number of pages in the pdf file
    """

    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)
    pages = fitz.open(pdf_dir)
    for page in pages:  # iterate through the pages
        image = page.get_pixmap(matrix=mat)  # render page to an image
        image.save(f"{work_dir}/page-{page.number}.png")
    return pages.page_count


class PDFReaderCustom(BaseReader):
    """PDF parser depending on if the pdf is flat or not."""

    image_loader: BaseReader

    def __init__(self, image_loader: BaseReader):
        """
        :param self: Represent the instance of the class
        :param image_loader: BaseReader: Pass the image_loader object to the class
        :return: An object of the class
        """
        self.image_loader = image_loader

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse file."""
        
        with open(file, "rb") as fp:
            # Create a PDF object
            pdf = pypdf.PdfReader(fp)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            # check if pdf content is flat or not
            if not num_pages > 0:
                raise Exception("PDF has no pages")
            examine_text = pdf.pages[0].extract_text()

            # Flat pdf (images) preprocessing to subdir with images for each pdf page:
            flat_pdf = True if examine_text == '' else False
            if flat_pdf:
                pdf_dir: Path = file
                work_dir: str = str(
                    pathlib.Path().resolve()
                ) + "/flat_pdf/{file_name}".format(
                    file_name=file.name.replace(file.suffix, "")
                )

                shutil.rmtree(
                    str(pathlib.Path().resolve()) + "/flat_pdf", ignore_errors=True
                )
                os.makedirs(work_dir)

                pdf_pages_count: int = convert_pdf_in_images(
                    pdf_dir=pdf_dir, work_dir=work_dir
                )
            
            if not pdf_pages_count == num_pages:
                raise Exception("Mismatch between page-count with pypdf and fitz")

            # pdf
            docs = []
            for page in range(num_pages):
                # Extract the text from the page
                if flat_pdf:
                    document = self.image_loader.load_data(
                        file=Path(work_dir + f"/page-{page}.png")
                    )
                    page_text = document[0].text
                else:
                    page_text = pdf.pages[page].extract_text()
                page_label = pdf.page_labels[page]

                metadata = {"page_label": page_label, "file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)

                docs.append(Document(text=page_text, metadata=metadata))
            return docs

