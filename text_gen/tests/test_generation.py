import unittest

from text_generator_2.settings import (
    MODEL_NAME,
    MODEL_PATH,
    MODEL_TYPE,
    TRAINING_TOP_K,
    TRAINING_TOP_P,
    VOCAB_SIZE,
)
from text_generator_2.text_gen.generation import generate_text
from text_generator_2.text_gen.models import GPT2Model


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.model = GPT2Model(MODEL_NAME, MODEL_TYPE, VOCAB_SIZE, MODEL_PATH)
        self.prompt = "What is the capital of"
        self.top_p = TRAINING_TOP_P
        self.top_k = TRAINING_TOP_K

    def test_generate_text(self):
        text = generate_text(self.model, self.prompt, self.top_p, self.top_k)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), len(self.prompt))

    def test_generate_text_with_invalid_prompt(self):
        self.prompt = None
        with self.assertRaises(ValueError) as context:
            generate_text(self.model, self.prompt, self.top_p, self.top_k)
        self.assertTrue("Prompt cannot be None or empty" in str(context.exception))

    def test_generate_text_with_invalid_top_p(self):
        self.top_p = 2
        with self.assertRaises(ValueError) as context:
            generate_text(self.model, self.prompt, self.top_p, self.top_k)
        self.assertTrue("top_p should be between 0 and 1" in str(context.exception))

    def test_generate_text_with_invalid_top_k(self):
        self.top_k = -1
        with self.assertRaises(ValueError) as context:
            generate_text(self.model, self.prompt, self.top_p, self.top_k)
        self.assertTrue("top_k should be greater than 0" in str(context.exception))
