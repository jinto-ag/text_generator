import os
import torch

# GPT-2 model settings
MODEL_NAME = "gpt2-large"
MODEL_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 50000
MODEL_PATH = os.path.join(os.getcwd(), "models", MODEL_NAME)

# Training settings
TRAINING_BATCH_SIZE = 64
TRAINING_EPOCHS = 10
TRAINING_MAX_LENGTH = 100
TRAINING_TOP_P = 0.9
TRAINING_TOP_K = 40
TRAINING_DATASET_PATH = os.path.join(os.getcwd(), "datasets", "training.txt")

# Evaluation settings
EVAL_DATASET_PATH = os.path.join(os.getcwd(), "datasets", "eval.txt")

# PDF text extraction settings
PDF_DIRECTORY = os.path.join(os.getcwd(), "pdfs")
