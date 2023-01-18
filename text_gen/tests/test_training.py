import unittest

from text_generator_2.settings import (
    MODEL_NAME,
    MODEL_PATH,
    MODEL_TYPE,
    TRAINING_BATCH_SIZE,
    TRAINING_DATASET_PATH,
    TRAINING_EPOCHS,
    TRAINING_MAX_LENGTH,
    VOCAB_SIZE,
)
from text_generator_2.text_gen.models import GPT2Model
from text_generator_2.text_gen.training import train_model


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.model = GPT2Model(MODEL_NAME, MODEL_TYPE, VOCAB_SIZE, MODEL_PATH)
        self.batch_size = TRAINING_BATCH_SIZE
        self.epochs = TRAINING_EPOCHS
        self.max_length = TRAINING_MAX_LENGTH
        self.dataset_path = TRAINING_DATASET_PATH

    def test_train_model(self):
        self.model.load_model()
        self.model.optimize_model()
        self.model = train_model(
            self.model, self.batch_size, self.epochs, self.max_length, self.dataset_path
        )
        self.assertIsNotNone(self.model)

    def test_train_model_with_invalid_batch_size(self):
        self.batch_size = -1
        with self.assertRaises(ValueError) as context:
            train_model(
                self.model,
                self.batch_size,
                self.epochs,
                self.max_length,
                self.dataset_path,
            )
        self.assertTrue("Batch size should be greater than 0" in str(context.exception))

    def test_train_model_with_invalid_epochs(self):
        self.epochs = -1
        with self.assertRaises(ValueError) as context:
            train_model(
                self.model,
                self.batch_size,
                self.epochs,
                self.max_length,
                self.dataset_path,
            )
            self.assertTrue("Epochs should be greater than 0" in str(context.exception))

    def test_train_model_with_invalid_max_length(self):
        self.max_length = -1
        with self.assertRaises(ValueError) as context:
            train_model(
                self.model,
                self.batch_size,
                self.epochs,
                self.max_length,
                self.dataset_path,
            )
        self.assertTrue("Max length should be greater than 0" in str(context.exception))

    def test_train_model_with_invalid_dataset_path(self):
        self.dataset_path = "/invalid/path/to/dataset"
        with self.assertRaises(FileNotFoundError) as context:
            train_model(
                self.model,
                self.batch_size,
                self.epochs,
                self.max_length,
                self.dataset_path,
            )
        self.assertTrue("No such file or directory" in str(context.exception))
