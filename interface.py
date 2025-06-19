from model_loader import ModelLoader
from chat_memory import ChatMemory
import os

class ChatbotInterface:
    """
    Manages the command-line interface for the chatbot, integrating
    the model loader and chat memory components.
    """
    def __init__(self, model_name: str = "microsoft/phi-2", max_memory_turns: int = 5): # CHANGED MODEL HERE
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
                prompt = self.chat_memory.get_conversation_history()
                # print(f"\n[DEBUG] Prompt fed to model:\n---START PROMPT---\n{prompt}\n---END PROMPT---\n")

                # Generate a response using the loaded Hugging Face model pipeline.
                # Parameters control the length, creativity, and stopping conditions of the generation.
                generation_output = self.generator(
                    prompt,
                    max_new_tokens=100,  # Reducing max tokens to encourage conciseness and prevent rambling.
                                         # Can be adjusted if answers are consistently too short.
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.6,     # Slightly lower temperature for less randomness/more focused answers.
                    top_k=50,
                    top_p=0.9,           # Slightly lower top_p for more focused token selection.
                    truncation=True,
                    pad_token_id=self.generator.tokenizer.eos_token_id,
                    eos_token_id=self.generator.tokenizer.eos_token_id
                )

                # Extract the generated text from the model's output.
                full_generated_text = generation_output[0]['generated_text']

                # The model generates the prompt + its response. We need to remove the original prompt
                # to get only the bot's newly generated part.
                bot_response_raw = full_generated_text.replace(prompt, "", 1).strip()

                # --- REVISED POST-PROCESSING FOR PHI-2 VERBOSITY ---
                # Define patterns that indicate the end of the desired response or a start of unwanted text.
                # The order here is important: we want to cut at the *earliest* unwanted pattern.
                stop_phrases = [
                    "\nUser:",          # Catches the start of a new user turn
                    "\nBot:",           # Catches the start of a new bot turn (if model repeats itself)
                    "\nAssistant:",     # Catches if the model shifts to an 'Assistant' persona
                    "\n\n",             # Catches explicit blank lines often indicating end of a thought block
                    "\nQuestion:",      # Catches patterns like "Question:..."
                    "\nAnswer:",        # Catches patterns like "Answer:..."
                    "\nImagine a universe", # Catches the specific rambling you observed
                    "\nThank you for",  # Catches parts of the polite but irrelevant rambling
                    "\nI need help with" # Catches parts of the technical issue scenario
                ]

                bot_response = bot_response_raw
                # Iterate through stop phrases and truncate the response if one is found.
                # We stop at the *first* match to get the most concise desired response.
                for phrase in stop_phrases:
                    if phrase in bot_response:
                        bot_response = bot_response.split(phrase)[0].strip()
                        break # Found a cutoff point, so stop looking for others.

                # If the bot's response still starts with "Bot:" or "Assistant:" (e.g., if it didn't generate a newline),
                # remove that prefix, as our display already adds "Bot:".
                if bot_response.lower().startswith("bot:"):
                    bot_response = bot_response[len("bot:"):].strip()

                # Final strip to remove any lingering leading/trailing whitespace.
                bot_response = bot_response.strip()

                # Fallback if the response becomes completely empty after cleaning.
                if not bot_response:
                    bot_response = "I'm sorry, I couldn't provide a clear answer for that. Can you please rephrase?"
                # --- END REVISED POST-PROCESSING ---

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
    chatbot = ChatbotInterface(model_name="microsoft/phi-2", max_memory_turns=5) # CHANGED MODEL HERE
    chatbot.start()