import openai
import openai_secret_manager


def generate_text(prompt: str) -> str:
    """
    Generate text using the GPT-2 model
    """
    secrets = openai_secret_manager.get_secrets("openai")
    openai.api_key = secrets["api_key"]
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]
