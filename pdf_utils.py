from pypdf import PdfReader
import logging


# Suppress noisy pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_pdf_text(pdf_source):

    reader = PdfReader(pdf_source)

    document_name = pdf_source.name

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text(
            extraction_mode="layout"
        )

        if text and text.strip():

            pages.append({
                "document": document_name,
                "page": page_number,
                "text": text.strip()
            })

    return pages