import os
import shutil
import unittest

from text_generator_2.text_extract.pdf_text_extraction import (
    extract_pdf_text_multithread,
    extract_text_from_pdf,
    get_pdf_files,
    split_dataset,
)


class TestPdfTextExtraction(unittest.TestCase):
    def setUp(self):
        # Create a test directory with pdf files
        self.test_dir = "./test_pdfs"
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "test1.pdf"), "w") as f:
            f.write("This is a test pdf file.")
        with open(os.path.join(self.test_dir, "test2.pdf"), "w") as f:
            f.write("This is another test pdf file.")
        with open(os.path.join(self.test_dir, "test3.pdf"), "w") as f:
            f.write("This is a test pdf file with a lot of text. " * 50)
        with open(os.path.join(self.test_dir, "test4.pdf"), "w") as f:
            f.write("This is another test pdf file with a lot of text. " * 50)

    def test_get_pdf_files(self):
        pdf_files = get_pdf_files(self.test_dir)
        self.assertEqual(len(pdf_files), 4)
        self.assertTrue(os.path.join(self.test_dir, "test1.pdf") in pdf_files)
        self.assertTrue(os.path.join(self.test_dir, "test2.pdf") in pdf_files)
        self.assertTrue(os.path.join(self.test_dir, "test3.pdf") in pdf_files)
        self.assertTrue(os.path.join(self.test_dir, "test4.pdf") in pdf_files)

    def test_extract_text_from_pdf(self):
        pdf_text = extract_text_from_pdf(os.path.join(self.test_dir, "test1.pdf"))
        self.assertEqual(pdf_text, "This is a test pdf file.")
        pdf_text = extract_text_from_pdf(os.path.join(self.test_dir, "test3.pdf"))
        self.assertEqual(
            len(pdf_text), len("This is a test pdf file with a lot of text. " * 50)
        )

    def test_split_dataset(self):
        pdf_texts = [
            "pdf1",
            "pdf2",
            "pdf3",
            "pdf4",
            "pdf5",
            "pdf6",
            "pdf7",
            "pdf8",
            "pdf9",
            "pdf10",
        ]
        train_texts, eval_texts = split_dataset(pdf_texts, 0.8)
        self.assertEqual(len(train_texts), 8)
        self.assertEqual(len(eval_texts), 2)
        train_texts, eval_texts = split_dataset(pdf_texts, 0.5)
        self.assertEqual(len(train_texts), 5)
        self.assertEqual(len(eval_texts), 5)

    def test_extract_pdf_text_multithread(self):
        pdf_texts = extract_pdf_text_multithread(self.test_dir)
        self.assertEqual(len(pdf_texts), 4)
        self.assertTrue("This is a test pdf file." in pdf_texts)
        self.assertTrue("This is another test pdf file." in pdf_texts)
        self.assertTrue(
            "This is a test pdf file with a lot of text. " * 50 in pdf_texts
        )
        self.assertTrue(
            "This is another test pdf file with a lot of text. " * 50 in pdf_texts
        )

    def tearDown(self):
        # Remove the test directory
        shutil.rmtree(self.test_dir)
