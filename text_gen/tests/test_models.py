import unittest

from text_generator_2.settings import MODEL_NAME, MODEL_PATH, MODEL_TYPE, VOCAB_SIZE
from text_generator_2.text_gen.models import GPT2Model


class TestModels(unittest.TestCase):
    def setUp(self):
        self.model = GPT2Model(MODEL_NAME, MODEL_TYPE, VOCAB_SIZE, MODEL_PATH)

    def test_load_model(self):
        self.model.load_model()
        self.assertIsNotNone(self.model.model)

    def test_optimize_model(self):
        self.model.load_model()
        self.model.optimize_model()
        self.assertIsNotNone(self.model.optimized_model)

    def test_load_model_with_invalid_path(self):
        self.model.model_path = "/invalid/path/to/model"
        with self.assertRaises(FileNotFoundError) as context:
            self.model.load_model()
        self.assertTrue("No such file or directory" in str(context.exception))
