# Local Command-Line Chatbot using Hugging Face

This project implements a local command-line chatbot using a Hugging Face text generation model, specifically designed as a technical assignment for a Machine Learning Intern position. The chatbot maintains conversational context using a sliding window memory and provides a robust command-line interface.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Model Used](#model-used)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Sample Interaction Examples](#sample-interaction-examples)
- [Code Structure](#code-structure)
- [Design Decisions](#design-decisions)
- [Evaluation Criteria Addressed](#evaluation-criteria-addressed)

## Project Overview

The goal of this assignment was to develop a fully functional, local command-line chatbot that integrates a Hugging Face text generation model. Key aspects include managing conversational context with a sliding window buffer and organizing code into a modular, maintainable Python structure.

## Features

-   **Local Model Inference:** Runs entirely on the user's machine (CPU or GPU).
-   **Hugging Face `pipeline`:** Utilizes `transformers.pipeline` for simplified text generation and tokenizer management.
-   **Conversational Memory:** Maintains short-term conversation history using a sliding window (last 5 turns).
-   **Modular Codebase:** Organized into `model_loader.py`, `chat_memory.py`, and `interface.py` for clarity and maintainability.
-   **Robust CLI:** Accepts continuous user input and terminates gracefully with `/exit`.
-   **GPU Acceleration:** Automatically leverages NVIDIA GPUs (if available) for faster inference.

## Model Used

-   **Name:** `microsoft/phi-1_5`
-   **Parameters:** 1.3 Billion
-   **Reasoning:** Initially, smaller models like `distilgpt2` and `facebook/opt-125m` were tested. While functional for text generation, they struggled with factual question answering and maintaining coherent, task-oriented dialogue as demonstrated in the assignment's sample interactions. `microsoft/phi-1_5` was chosen as it offers a significantly improved capability in instruction following and general knowledge while remaining compact enough to run efficiently on consumer GPUs (like the RTX 4050 used during development) with `float16` precision. It also utilizes the safer `safetensors` format, avoiding `torch.load` vulnerabilities.

## Setup Instructions

1.  **Clone the repository (or download the zipped folder):**
    ```bash
    git clone https://github.com/your-username/CLI-Chatbot.git
    cd CLI-Chatbot
    ```
    (If using a zipped folder, unzip it and navigate into the `CLI-Chatbot` directory.)

2.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Windows (Command Prompt):**
        ```bash
        venv\Scripts\activate.bat
        ```
    *   **On Windows (PowerShell):**
        ```bash
        venv\Scripts\Activate.ps1
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    (You should see `(venv)` at the start of your terminal prompt.)

4.  **Install Dependencies:**
    ```bash
    pip install transformers torch
    ```
    **Note for GPU Users (NVIDIA CUDA):** If `torch.cuda.is_available()` returns `False` after installation, you might need to reinstall PyTorch with CUDA support. For CUDA 12.1 (common for RTX GPUs):
    ```bash
    pip uninstall torch -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

## How to Run

1.  **Ensure your virtual environment is active** (as per setup instructions).
2.  **Run the main interface script:**
    ```bash
    python interface.py
    ```

The first time you run the chatbot, the `microsoft/phi-1_5` model (approx. 2.8 GB) and its tokenizer will be downloaded and cached. This may take several minutes depending on your internet speed. Subsequent runs will load much faster from cache.

## Sample Interaction Examples

```bash
Initializing chatbot. Please wait...
[ModelLoader] Loading model 'microsoft/phi-1_5'...
Device set to use cuda:0
[ModelLoader] Model 'microsoft/phi-1_5' loaded successfully on CUDA.

==================================================
 Welcome to the Local Hugging Face Chatbot!
 Type your message and press Enter. Type '/exit' to quit.
==================================================

User: Hi! How are you?
Bot: Hello! I'm doing well, thank you. How about you?
User: What is the language spoken in Portugal?
Bot: Portuguese is the official language of Portugal. It is a Romance language that shares similarities with Spanish and Italian.
User: What about India?
Bot: In India, the primary language spoken is Hindi. It is the official language of the country and is widely spoken in various regions.
User: Do you know about Brazil?
Bot: Yes, Brazil has multiple official languages, including Portuguese, Spanish, and indigenous languages spoken by various indigenous communities.
User: List all the languages spoken in India?
Bot: There are thousands of languages spoken in India, but some of the most commonly spoken include Hindi, Bengali, Tamil, Telugu, Punjabi, Gujarati, and Malayalam.
User: /exit
Exiting chatbot. Goodbye!
```

## Code Structure

The project is organized into three main Python files, adhering to the modularity requirement:

-   **`model_loader.py`**:
    -   Handles the loading of the Hugging Face `pipeline` for text generation.
    -   Automatically detects and utilizes GPU (CUDA) if available.
    -   Configures the model for efficient inference (e.g., `torch.float16` for GPU).
-   **`chat_memory.py`**:
    -   Manages the conversation history using a `collections.deque`.
    -   Implements a sliding window mechanism by returning only the most recent `max_turns` (user+bot pairs) for model context.
    -   Formats the history into a conversational string (e.g., "User: ...\nBot: ...\nBot:") suitable for language models.
-   **`interface.py`**:
    -   The main entry point for the chatbot application.
    -   Contains the CLI loop for continuous user interaction.
    -   Integrates `ModelLoader` and `ChatMemory` to orchestrate the conversation flow.
    -   Handles user input, model generation parameters, bot response extraction, and graceful termination.

## Design Decisions

1.  **Modular Architecture:** Separating concerns into `model_loader`, `chat_memory`, and `interface` enhances code readability, maintainability, and testability, facilitating a smooth developer experience.
2.  **Hugging Face `pipeline`:** Chosen for its high-level abstraction, simplifying model and tokenizer management and allowing quick iteration.
3.  **`collections.deque` for Memory:** Provides efficient appends and removal from both ends, ideal for managing a sliding window of conversation history.
4.  **`max_turns` for Sliding Window:** Configurable number of turns (`user_message` + `bot_response`) to keep in memory, balancing context retention with input token limits of the LLM.
5.  **Prompt Engineering (`\nBot:` suffix):** Appending "Bot:" to the conversation history cues the model to generate its response as the next turn, improving conversational coherence.
6.  **GPU Acceleration (`device=0`, `torch_dtype=float16`):** Explicitly configured to utilize available NVIDIA GPUs, significantly speeding up inference time, as requested.
7.  **`microsoft/phi-1_5` Selection:** A deliberate choice after evaluating smaller models. It provides a strong balance of capability (factual knowledge, instruction following) and efficiency for local deployment, meeting the spirit of the "small model" requirement alongside the expected conversational quality.
8.  **Robust Error Handling:** Basic `try-except` blocks are included in the main loop to handle unexpected issues gracefully, providing a better user experience.

## Evaluation Criteria Addressed

This project directly addresses the following evaluation criteria:

-   **Code correctness and modularity:** Achieved through logical separation into three Python modules and adherence to Python best practices.
-   **Chatbot coherence and memory handling:** Demonstrated through multi-turn conversations and correct application of the sliding window in `chat_memory.py`. The `phi-1_5` model ensures high coherence.
-   **Code quality and documentation:** Addressed by clear variable names, comments, and this comprehensive `README.md`.
-   **Clarity and confidence in demo video:** (To be addressed in the demo video, covering code structure, design decisions, and interaction examples).