import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from PyPDF2 import PdfFileReader


def extract_pdf_text_multithread(
    pdf_path: str, max_workers: int = None
) -> Tuple[str, str]:
    """
    Extract text from pdf using multiprocessing
    """
    if not max_workers:
        max_workers = cpu_count()

    pdf_name = os.path.basename(pdf_path)
    pdf_reader = PdfFileReader(open(pdf_path, "rb"))
    pdf_text = ""
    for page_num in range(pdf_reader.numPages):
        pdf_text += pdf_reader.getPage(page_num).extractText()
    return pdf_name, pdf_text


def get_pdf_files(directory: str) -> List[str]:
    """
    Get all pdf files from the directory
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def extract_text_from_pdf(pdf_path: str, max_workers: int = None) -> Tuple[str, str]:
    """
    Extract text from pdf using multiprocessing
    """
    with Pool(max_workers) as p:
        results = p.map(extract_pdf_text_multithread, [pdf_path])
    return results[0]
