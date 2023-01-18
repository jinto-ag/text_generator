import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from text_generator_2 import settings

class TextGenerator:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(settings.MODEL_NAME).to(settings.MODEL_TYPE)
        
    def generate_text(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.5):
        """
        Generate text using the GPT-2 model
        """
        input_ids = torch.tensor(self.tokenizer.encode(prompt, return_tensors="pt")).to(settings.MODEL_TYPE)
        output = self.model.generate(input_ids=input_ids, max_length=max_tokens, temperature=temperature)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def save_model(self, model_path: str):
        """
        Save the model to disk
        """
        torch.save(self.model.state_dict(), model_path)
        
    def load_model(self, model_path: str):
        """
        Load the model from disk
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(settings.MODEL_TYPE)
        self.model.eval()
