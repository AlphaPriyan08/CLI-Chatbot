from collections import deque

class ChatMemory:
    """
    Manages the conversation history for the chatbot using a sliding window mechanism.
    It stores individual messages and formats them into a coherent string for the language model.
    """
    def __init__(self, max_turns: int = 5):
        """
        Initializes the ChatMemory with a maximum number of conversation turns to remember.
        A 'turn' is typically defined as one user message followed by one bot message.
        
        Args:
            max_turns (int): The maximum number of user-bot message pairs to retain in memory.
                             This translates to roughly `max_turns * 2` individual messages.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.max_turns = max_turns
        # Using a deque (double-ended queue) for efficient addition and retrieval
        # of messages from either end, suitable for sliding window implementation.
        self.history = deque()

    def add_message(self, speaker: str, text: str):
        """
        Adds a single message (from either User or Bot) to the conversation history.
        
        Args:
            speaker (str): The sender of the message (e.g., "User", "Bot").
            text (str): The content of the message.
        """
        self.history.append({"speaker": speaker, "text": text})

        # The actual sliding window truncation (removing older messages) is
        # primarily handled when `get_conversation_history` is called, ensuring
        # the model always receives the most recent and relevant context.
        # No explicit popping here to maintain flexibility for history retrieval.
        pass

    def get_conversation_history(self) -> str:
        """
        Formats the most recent conversation history into a single string
        using an instruction-following format suitable for models like Phi-1.5.
        The sliding window logic is applied here.
        """
        recent_messages = list(self.history)[-self.max_turns * 2:]

        # Build the internal conversation dialogue using the User: / Bot: format.
        inner_conversation_parts = []
        for msg in recent_messages:
            inner_conversation_parts.append(f"{msg['speaker']}: {msg['text']}")

        # Join the individual conversation lines into a single string.
        conversation_string = "\n".join(inner_conversation_parts)

        # Wrap the entire conversation history within the "Instruct:" and "Output:" format.
        # This explicitly tells the model it's an instruction to provide an output based on history.
        formatted_history = f"Instruct: {conversation_string}\nOutput:"
        
        return formatted_history

    def clear_history(self):
        """
        Clears the entire conversation history, resetting the chatbot's memory.
        """
        self.history.clear()
        print("[ChatMemory] Conversation history cleared.")
