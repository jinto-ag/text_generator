import torch
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

from text_generator_2 import settings


def train_model(dataset_path: str):
    """
    Train the GPT-2 model on a dataset
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(settings.MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(settings.MODEL_NAME)

    # Prepare dataset
    dataset = torch.load(dataset_path)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataset) * settings.TRAINING_EPOCHS,
    )

    for epoch in range(settings.TRAINING_EPOCHS):
        # Initialize hidden states
        model.train()
        total_loss = 0

        for i, data in enumerate(dataset):
            optimizer.zero_grad()
            input_ids = data["input_ids"].to(settings.MODEL_TYPE)
            attention_mask = data["attention_mask"].to(settings.MODEL_TYPE)
            labels = data["labels"].to(settings.MODEL_TYPE)

            # Forward pass
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                max_length=settings.TRAINING_MAX_LENGTH,
                top_p=settings.TRAINING_TOP_P,
                top_k=settings.TRAINING_TOP_K,
            )
            loss, logits = outputs[:2]
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Print loss and perplexity
        avg_loss = total_loss / len(dataset)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"Epoch: {epoch+1} Loss: {avg_loss:.4f} Perplexity: {perplexity:.4f}")
