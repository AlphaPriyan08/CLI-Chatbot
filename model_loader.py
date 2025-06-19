from transformers import pipeline
import torch

class ModelLoader:
    """
    A class to load and manage the Hugging Face text generation pipeline.
    """
    def __init__(self, model_name: str = "microsoft/phi-1_5"): # CHANGED MODEL HERE
        self.model_name = model_name
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{self.__class__.__name__}] Using device: {self.device.upper()}")

    def load_model(self):
        """
        Loads the Hugging Face text generation pipeline.
        This will download the model and tokenizer if not already cached.
        """
        if self.pipeline is None:
            print(f"[{self.__class__.__name__}] Loading model '{self.model_name}'...")
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    # low_cpu_mem_usage=True, 
                    trust_remote_code=True 
                )
                print(f"[{self.__class__.__name__}] Model '{self.model_name}' loaded successfully on {self.device.upper()}.")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error loading model: {e}")
                print("Please ensure the model name is correct and you have an internet connection.")
                print("If loading a Phi model, ensure `trust_remote_code=True` is set in the pipeline call.")
                self.pipeline = None # Reset pipeline on failure
        return self.pipeline

    def get_model_name(self):
        """
        Returns the name of the loaded model.
        """
        return self.model_name

# Example usage (for testing purposes, will be removed later)
if __name__ == "__main__":
    # Create an instance of ModelLoader
    loader = ModelLoader(model_name="microsoft/phi-1_5")

    # Load the model
    # The first time you run this, it will download the model, which might take a few minutes.
    # Subsequent runs will load from cache, which is much faster.
    generator = loader.load_model()

    if generator:
        print("\n[Test] Generating a simple text snippet...")
        prompt = "Hello, my name is "
        # Generate text with some basic parameters
        # max_new_tokens: The maximum number of tokens to generate in the response.
        # num_return_sequences: How many different sequences to generate (we want 1 for a chatbot).
        # truncation: If the input prompt is too long for the model's context window, truncate it.
        # pad_token_id and eos_token_id: Essential for text generation with GPT-like models.
        # They prevent warnings and help the model know when to stop generating.
        # We get them from the tokenizer (implicitly handled by pipeline).
        # For 'distilgpt2', the 'eos_token_id' is usually 50256 (end-of-sequence token).
        # pad_token_id is also often the same for GPT-like models when no explicit pad token is defined.
        generated_text = generator(
            prompt,
            max_new_tokens=50,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=generator.tokenizer.eos_token_id, # Use EOS as PAD
            eos_token_id=generator.tokenizer.eos_token_id
        )

        print("\n--- Generated Text ---")
        print(generated_text[0]['generated_text'])
        print("----------------------")
    else:
        print("\n[Test] Model loading failed. Cannot generate text.")