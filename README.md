# Building and Deploying a Gradio UI on Hugging Face Spaces

## Overview

This repository contains the code of the "Building and Deploying a Gradio UI on Hugging Face Spaces" lesson of the "From Beginner to Advanced LLM Developer" course.

## Setup

1. Clone the repository.

```bash
git clone git@github.com:towardsai/ai-tutor-gradio-lesson.git
cd ai-tutor-gradio-lesson
```

2. Create a `.env` file and add there your OpenAI API key. Its content should be something like:

```bash
OPENAI_API_KEY="sk-..."
```

3. Create a local virtual environment, for example using the `venv` module. Then, activate it.

```bash
python -m venv venv
source venv/bin/activate
```

4. Install the dependencies.

```bash
pip install -r requirements.txt
```

5. Launch the Gradio app.

```bash
python app.py
```