from transformers import pipeline
import torch

class ModelLoader:
    """
    A class to encapsulate the loading and management of a Hugging Face
    text generation pipeline.
    """
    def __init__(self, model_name: str = "microsoft/phi-2"):
        # smaller models like distilgpt2 or OPT-125m struggled with factual question answering and maintaining coherence 
        self.model_name = model_name
        self.pipeline = None
        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[{self.__class__.__name__}] Using device: {self.device.upper()}")

    def load_model(self):
        """
        Loads the Hugging Face text generation pipeline for the specified model.
        This function handles model and tokenizer download/caching and device placement.
        """
        if self.pipeline is None:
            print(f"[{self.__class__.__name__}] Loading model '{self.model_name}'...")
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    # Assign to the first GPU (0) if CUDA is available, otherwise use CPU (-1).
                    device=0 if self.device == "cuda" else -1,
                    # Use float16 precision on GPU for faster inference and reduced memory usage.
                    # Fallback to float32 on CPU.
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    # Required for some custom models (like Microsoft Phi) that include custom code.
                    trust_remote_code=True
                )
                print(f"[{self.__class__.__name__}] Model '{self.model_name}' loaded successfully on {self.device.upper()}.")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error loading model: {e}")
                print("Please ensure the model name is correct, you have an internet connection, and necessary dependencies.")
                print("If loading a Phi model, ensure `trust_remote_code=True` is set in the pipeline call.")
                self.pipeline = None # Reset pipeline on failure
        return self.pipeline

    def get_model_name(self):
        """
        Returns the name of the model configured for loading.
        """
        return self.model_name
