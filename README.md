# Towards AI 🤖: An AI Question-Answering Bot

## Overview

**Towards AI 🤖** is a question-answering bot designed to assist students with queries related to Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL). It leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques to provide insightful answers, utilizing a vector database for efficient retrieval of knowledge.

## Features

- AI, ML, and DL question-answering capabilities.
- Integration with ChromaDB for persistent storage.
- Utilizes OpenAI's models for generating responses.
- Gradio interface for easy interaction.
- Memory management for maintaining conversation context.

## Requirements

Make sure you have installed the dependencies from requirements.txt file.

```bash
pip install -r requirements.txt
```

## Setup

1. **Clone the Repository**
    
   ```bash
   git clone https://huggingface.co/yourusername/towards-ai
   cd towards-ai
   ```

2. **Environment Variables**

    Create a .env file in the project root and set the following variables:
    ```bash
    OPENAI_API_KEY=
    LOGFIRE_TOKEN=
    COHERE_API_KEY=
    MONGODB_URI= 
    DB_NAME=ai_tutor_knowledge
   ```
3. **Download the Vector Database**
   
   The bot requires a pre-trained vector database. If it doesn't exist locally, it will automatically download it from Hugging Face Hub when you run the code.

4. **Usage**

    To start the chatbot, run the following command:
    ```bash
    python app.py
   ```

5. **Gradio Interface**

    Once the application is running, you can access the chatbot interface at http://localhost:7860.

6. **Interacting with the Bot**

   You can ask the bot any question related to AI, ML, or DL. The bot is designed to provide clear, complete answers based on its knowledge base.