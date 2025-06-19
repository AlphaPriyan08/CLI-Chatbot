# interface.py

from model_loader import ModelLoader
from chat_memory import ChatMemory
import os # For clear screen functionality (optional but nice)

class ChatbotInterface:
    """
    Manages the command-line interface for the chatbot, integrating
    the model loader and chat memory components.
    """
    def __init__(self, model_name: str = "microsoft/phi-1_5", max_memory_turns: int = 5):
        self.model_loader = ModelLoader(model_name=model_name)
        self.chat_memory = ChatMemory(max_turns=max_memory_turns)
        self.generator = None # Will store the loaded pipeline

    def _clear_screen(self):
        """Clears the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def start(self):
        """
        Starts the main chatbot loop.
        """
        self._clear_screen()
        print("Initializing chatbot. Please wait...")

        # 1. Load the model
        self.generator = self.model_loader.load_model()
        if not self.generator:
            print("Chatbot initialization failed. Exiting.")
            return

        print("\n" + "="*50)
        print(" Welcome to the Local CLI Chatbot!")
        print(" Type your message and press Enter. Type '/exit' to quit.")
        print("="*50 + "\n")

        # 2. Main CLI loop
        while True:
            try:
                user_input = input("User: ").strip()

                if user_input.lower() == "/exit":
                    print("Exiting chatbot. Goodbye!")
                    break

                if not user_input: # Handle empty input
                    continue

                # Add user message to memory
                self.chat_memory.add_message("User", user_input)

                # Get conversation history for the model
                # The "Bot:" suffix in get_conversation_history prompts the model
                # to generate the next response as if it's the bot speaking.
                prompt = self.chat_memory.get_conversation_history()
                # print(f"\n[DEBUG] Prompt fed to model:\n{prompt}\n") # Uncomment for debugging prompt

                # Generate bot response
                # Key parameters for text generation:
                #   max_new_tokens: Max tokens the model will generate for the response.
                #   num_return_sequences: We only want one coherent response.
                #   truncation: True if the combined prompt + history might exceed model's max input length.
                #   pad_token_id/eos_token_id: Essential for proper generation termination and avoiding warnings.
                #   do_sample: True enables sampling (more creative, less deterministic).
                #   temperature: Controls randomness (lower = more focused, higher = more diverse).
                #   top_k: Filters out low probability tokens (good for creativity without going off rails).
                #   top_p: Nucleus sampling (another way to filter probabilities).
                #   repetition_penalty: Discourages the model from repeating itself.
                #   no_repeat_ngram_size: Prevents repeating n-grams (e.g., repeating "the the").
                # Adjust these for better quality!
                # For `distilgpt2`, the `pad_token_id` is typically the same as `eos_token_id` (50256).
                # If you get warnings about pad_token_id, ensure it's explicitly set.
                generation_output = self.generator(
                    prompt,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    truncation=True,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                    eos_token_id=self.generator.tokenizer.eos_token_id
                    # Ensure no other arguments like low_cpu_mem_usage, torch_dtype, trust_remote_code are here
                )
                # Extract the generated text
                full_generated_text = generation_output[0]['generated_text']

                # The model generates text based on the *entire* prompt.
                # We need to extract only the *new* bot response.
                # It usually generates `prompt` + `Bot: <actual_response>`.
                bot_response_raw = full_generated_text.replace(prompt, "", 1).strip()

                # Post-processing: sometimes models generate extra text or incomplete sentences.
                # Remove any leftover "User:" or "Bot:" from subsequent turns if model hallucinates.
                # Stop at the first newline or colon if it's a short response, or if another speaker is implied.
                if "\nUser:" in bot_response_raw:
                    bot_response = bot_response_raw.split("\nUser:")[0].strip()
                elif "\nBot:" in bot_response_raw:
                    bot_response = bot_response_raw.split("\nBot:")[0].strip()
                elif ":" in bot_response_raw and bot_response_raw.split(':')[0].strip().istitle():
                    # Handle cases like "Bot: Hi there." or "Assistant: How can I help?"
                    # If it starts with "Bot:", remove it, as our prompt already includes it.
                    if bot_response_raw.lower().startswith("bot:"):
                         bot_response = bot_response_raw[len("bot:"):].strip()
                    else:
                        bot_response = bot_response_raw # Take the whole thing
                else:
                    bot_response = bot_response_raw

                # Add bot response to memory
                self.chat_memory.add_message("Bot", bot_response)

                # Print bot response
                print(f"Bot: {bot_response}")

            except KeyboardInterrupt:
                print("\nExiting chatbot. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Please try again or type '/exit' to quit.")

# Main execution block
if __name__ == "__main__":
    # You can customize model_name and max_memory_turns here
    chatbot = ChatbotInterface(model_name="microsoft/phi-1_5", max_memory_turns=5)
    chatbot.start()