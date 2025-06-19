# interface.py

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
                # The `get_conversation_history` method now formats it as "Instruct: ...\nOutput:".
                prompt = self.chat_memory.get_conversation_history()
                # print(f"\n[DEBUG] Prompt fed to model:\n{prompt}\n") # Uncomment for debugging prompt

                # Generate a response using the loaded Hugging Face model pipeline.
                # Parameters control the length, creativity, and stopping conditions of the generation.
                generation_output = self.generator(
                    prompt,
                    max_new_tokens=200,  # Increased to allow for longer, more complete answers.
                    num_return_sequences=1, # We only need one response from the model.
                    do_sample=True,      # Enables sampling for more varied and human-like responses (vs. deterministic).
                    temperature=0.7,     # Adjusted slightly higher for more natural variety, can tune as needed.
                    top_k=50,            # Considers only the top K most probable next tokens.
                    top_p=0.95,          # Nucleus sampling: considers tokens whose cumulative probability exceeds p.
                    truncation=True,     # Ensures input prompt fits model's maximum context length by truncating older parts.
                    pad_token_id=self.generator.tokenizer.eos_token_id, # Tells the model what token to use for padding.
                    eos_token_id=self.generator.tokenizer.eos_token_id # Tells the model when to stop generating (end-of-sequence token).
                )

                # Extract the generated text from the model's output.
                full_generated_text = generation_output[0]['generated_text']

                # --- REFINED POST-PROCESSING FOR PHI-1.5 ---
                # Phi models often regenerate the "Instruct: ...\nOutput:" part or similar structures.
                # We need to extract only the actual bot response after the *last* "Output:".
                bot_response_raw = ""
                last_output_idx = full_generated_text.rfind("Output:")
                if last_output_idx != -1:
                    # Extract everything after the last "Output:" occurrence.
                    bot_response_raw = full_generated_text[last_output_idx + len("Output:"):].strip()
                else:
                    # Fallback: if "Output:" isn't found, assume model just started generating after the prompt.
                    # This might happen if the prompt was short and not reproduced.
                    bot_response_raw = full_generated_text.replace(prompt, "", 1).strip()

                # Define explicit patterns that indicate the end of a desired response
                # or a shift to an undesired format (like exercises or new speaker turns).
                stop_phrases = [
                    "\nUser:",          # If the model hallucinates a new user turn
                    "\nBot:",           # If the model hallucinates a new bot turn (old format)
                    "\nInstruct:",      # If the model hallucinates a new instruction
                    "\nQuestion:",      # Common start of a new question
                    "\nAnswer:",        # Common start of an answer
                    "\nExercise",       # Catches the "Exercise X:" pattern
                    "\n1.", "\n2.", "\n-", # Catches numbered/bulleted lists starting new ideas
                    "\n\n",             # Double newline often indicates end of thought.
                ]

                # Iterate through stop phrases and truncate the response if one is found.
                bot_response = bot_response_raw
                for phrase in stop_phrases:
                    if phrase in bot_response:
                        bot_response = bot_response.split(phrase)[0].strip()
                        break # Stop at the first encountered stop phrase

                # Final cleanup: remove any leading/trailing whitespace that might remain.
                bot_response = bot_response.strip()

                # Fallback if the response becomes empty after cleaning (e.g., model only generated a stop token).
                if not bot_response:
                    bot_response = "I'm sorry, I couldn't generate a clear response for that. Can you please rephrase?"
                # --- END REFINED POST-PROCESSING ---

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