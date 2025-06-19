from collections import deque

class ChatMemory:
    """
    Manages the conversation history using a sliding window.
    Stores messages as tuples (speaker, text).
    """
    def __init__(self, max_turns: int = 5):
        """
        Initializes the ChatMemory with a maximum number of turns to remember.
        Args:
            max_turns (int): The maximum number of user/bot message pairs to keep in memory.
                             A turn consists of one user message and one bot message.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.max_turns = max_turns
        self.history = deque() # Use deque for efficient appends and pops from both ends

    def add_message(self, speaker: str, text: str):
        """
        Adds a message to the conversation history.
        Args:
            speaker (str): The sender of the message (e.g., "User", "Bot").
            text (str): The content of the message.
        """
        # A simple way to manage turns: assume user and bot messages come in pairs.
        # This implementation simplifies by treating each message individually for the deque,
        # but the `get_conversation_history` method will handle the 'turn' concept
        # by only returning the last N messages, which implicitly covers N turns if balanced.
        self.history.append({"speaker": speaker, "text": text})

        # Implement sliding window: If we have more than (max_turns * 2) messages
        # (max_turns user messages + max_turns bot messages),
        # remove the oldest user message and bot message pair.
        # This keeps the buffer size to roughly max_turns * 2
        # However, it's simpler to just slice the last N total messages
        # as the context for the LM
        if len(self.history) > self.max_turns * 2: # Keep at most max_turns of (User + Bot) pairs
            # Remove the oldest messages until we are within the window
            # A more robust turn-based window would track (user, bot) pairs.
            # For simplicity, we will just take the last N * 2 individual messages when retrieving.
            # For this task, a simple slice of the 'deque' for the last N messages for context is sufficient.
            # So, the explicit popping for "sliding window" happens when *retrieving* the history.
            # Let's refine this to be purely based on `get_conversation_history` for simplicity and direct relevance to LM input size.
            # For `distilgpt2`, less strict window management here is fine; the model's max input length will handle truncation.
            pass # We'll manage the window when we *get* the history for the model.

    def get_conversation_history(self) -> str:
        """
        Formats the conversation history into a single string suitable for the language model.
        The format will be "User: <message>\nBot: <response>\nUser: <message>..."
        Applies the sliding window here by only considering the last `max_turns * 2` messages.
        Returns:
            str: The formatted conversation history string.
        """
        # We need to consider up to `max_turns` of User-Bot pairs.
        # So, we retrieve up to `max_turns * 2` individual messages.
        # deque.maxlen could be used if we strictly wanted to enforce it on `add_message`.
        # For simplicity and to allow flexible formatting, we manage it here.
        recent_messages = list(self.history)[-self.max_turns * 2:] # Get the last N * 2 messages

        formatted_history_parts = []
        for msg in recent_messages:
            formatted_history_parts.append(f"{msg['speaker']}: {msg['text']}")

        # Add "Bot:" at the end to prompt the model to generate the bot's response
        # based on the preceding context.
        # This is a common pattern for text generation models in a conversational setting.
        return "\n".join(formatted_history_parts) + "\nBot:"

    def clear_history(self):
        """
        Clears the entire conversation history.
        """
        self.history.clear()
        print("[ChatMemory] Conversation history cleared.")

# Example Usage (for testing purposes)
if __name__ == "__main__":
    memory = ChatMemory(max_turns=3) # Remember last 3 turns (6 messages total)

    print("--- Adding messages ---")
    memory.add_message("User", "Hello there!")
    memory.add_message("Bot", "Hi! How can I help you?")
    print(f"Current history (1 turn):\n{memory.get_conversation_history()}")

    memory.add_message("User", "What is the capital of France?")
    memory.add_message("Bot", "The capital of France is Paris.")
    print(f"\nCurrent history (2 turns):\n{memory.get_conversation_history()}")

    memory.add_message("User", "And what about Italy?")
    memory.add_message("Bot", "The capital of Italy is Rome.")
    print(f"\nCurrent history (3 turns):\n{memory.get_conversation_history()}")

    memory.add_message("User", "What about Germany?") # This will cause the first turn to drop
    memory.add_message("Bot", "The capital of Germany is Berlin.")
    print(f"\nCurrent history (4 turns, should show last 3):\n{memory.get_conversation_history()}")

    print("\n--- Clearing history ---")
    memory.clear_history()
    print(f"History after clearing:\n'{memory.get_conversation_history()}'") # Should be just "Bot:"