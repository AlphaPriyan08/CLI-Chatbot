from model_loader import ModelLoader
from chat_memory import ChatMemory
import os

class ChatbotInterface:
    """
    Manages the command-line interface for the chatbot, integrating
    the model loader and chat memory components.
    """
    def __init__(self, model_name: str = "microsoft/phi-1_5", max_memory_turns: int = 5):
        # Initialize the model loader responsible for fetching and preparing the LLM.
        self.model_loader = ModelLoader(model_name=model_name)
        # Initialize the chat memory to store and manage conversation history.
        self.chat_memory = ChatMemory(max_turns=max_memory_turns)
        self.generator = None # Placeholder for the loaded Hugging Face pipeline

    def _clear_screen(self):
        """Clears the terminal screen for a cleaner user experience."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def start(self):
        """
        Starts the main chatbot loop, handling user input, model generation,
        and conversational memory.
        """
        self._clear_screen()
        print("Initializing chatbot. Please wait...")

        # Load the text generation model using ModelLoader.
        # This will download the model the first time it's run.
        self.generator = self.model_loader.load_model()
        if not self.generator:
            print("Chatbot initialization failed. Exiting.")
            return

        print("\n" + "="*50)
        print(" Welcome to the Local CLI Chatbot!")
        print(" Type your message and press Enter. Type '/exit' to quit.")
        print("="*50 + "\n")

        # Main conversational loop.
        while True:
            try:
                user_input = input("User: ").strip()

                # Check for the exit command to gracefully terminate the chatbot.
                if user_input.lower() == "/exit":
                    print("Exiting chatbot. Goodbye!")
                    break

                # Skip processing if the user enters an empty line.
                if not user_input:
                    continue

                # Add the user's message to the conversation history.
                self.chat_memory.add_message("User", user_input)

                # Retrieve the formatted conversation history from memory.
                # The `get_conversation_history` method includes the "Bot:" suffix,
                # which acts as a prompt to encourage the model to generate a response.
                prompt = self.chat_memory.get_conversation_history()

                # Generate a response using the loaded Hugging Face model pipeline.
                # Parameters control the length, creativity, and stopping conditions of the generation.
                generation_output = self.generator(
                    prompt,
                    max_new_tokens=100,  # Maximum number of new tokens (words/subwords) the bot will generate.
                    num_return_sequences=1, # We only need one response from the model.
                    do_sample=True,      # Enables sampling for more varied and human-like responses (vs. deterministic).
                    temperature=0.7,     # Controls randomness; lower for more focused, higher for more creative.
                    top_k=50,            # Considers only the top K most probable next tokens.
                    top_p=0.95,          # Nucleus sampling: considers tokens whose cumulative probability exceeds p.
                    truncation=True,     # Ensures input prompt fits model's maximum context length by truncating older parts.
                    pad_token_id=self.generator.tokenizer.eos_token_id, # Tells the model what token to use for padding.
                    eos_token_id=self.generator.tokenizer.eos_token_id # Tells the model when to stop generating (end-of-sequence token).
                )

                # Extract the generated text from the model's output.
                full_generated_text = generation_output[0]['generated_text']

                # The model generates the prompt + its response. We need to remove the original prompt
                # to get only the bot's newly generated part.
                bot_response_raw = full_generated_text.replace(prompt, "", 1).strip()

                # Post-process the raw bot response to clean up any unwanted prefixes or truncated dialogue
                # that models sometimes generate.
                if "\nUser:" in bot_response_raw:
                    bot_response = bot_response_raw.split("\nUser:")[0].strip()
                elif "\nBot:" in bot_response_raw:
                    bot_response = bot_response_raw.split("\nBot:")[0].strip()
                elif ":" in bot_response_raw and bot_response_raw.split(':')[0].strip().istitle():
                    if bot_response_raw.lower().startswith("bot:"):
                         bot_response = bot_response_raw[len("bot:"):].strip()
                    else:
                        bot_response = bot_response_raw
                else:
                    bot_response = bot_response_raw

                # Add the bot's (cleaned) response to the conversation history.
                self.chat_memory.add_message("Bot", bot_response)

                # Display the bot's response to the user.
                print(f"Bot: {bot_response}")

            except KeyboardInterrupt:
                # Handle Ctrl+C for a graceful exit.
                print("\nExiting chatbot. Goodbye!")
                break
            except Exception as e:
                # Catch any unexpected errors during the conversation loop.
                print(f"An error occurred: {e}")
                print("Please try again or type '/exit' to quit.")

# Entry point for running the chatbot directly from the script.
if __name__ == "__main__":
    # Initialize the chatbot interface with the desired model and memory size.
    chatbot = ChatbotInterface(model_name="microsoft/phi-1_5", max_memory_turns=5)
    chatbot.start()