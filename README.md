---
title: AI Tutor
emoji: 💡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.26.0"
python_version: "3.12"
app_file: app.py
pinned: false
datasets:
  - towardsai-tutors/full-stack-ai-engineering-data
---

# Building and Deploying a Gradio UI on Hugging Face Spaces

## Overview

This repository contains the code of the "Building and Deploying a Gradio UI on Hugging Face Spaces" lesson of the "From Beginner to Advanced LLM Developer" course.

The app is the deployable version of the RAG AI Tutor built across the course notebooks: it loads a prebuilt Chroma vector store of the course knowledge base (788 documents, 7,428 chunks) from the [course dataset](https://huggingface.co/datasets/towardsai-tutors/full-stack-ai-engineering-data), retrieves and generates with [`tai-aitutor`](https://pypi.org/project/tai-aitutor/) (the course toolkit), keeps conversational memory with the hand-rolled summary strategy from the Section 9 memory lesson, and streams answers into a Gradio chat UI.

## Setup

1. Clone the repository.

```bash
git clone git@github.com:jaiganesan-n/AI_Tutor_Chatbot.git
cd AI_Tutor_Chatbot
```

2. Create a `.env` file with your Gemini API key (the course-default provider):

```bash
GOOGLE_API_KEY="..."
```

Only needed if you switch providers: `OPENAI_API_KEY`.

3. Create a virtual environment with **Python 3.12 or newer** (the `tai-aitutor` toolkit requires it), then activate it.

```bash
python3.12 -m venv venv
source venv/bin/activate
```

4. Install the dependencies.

```bash
pip install -r requirements.txt
```

5. Launch the Gradio app. On first run it downloads the prebuilt vector store (~100 MB, one time) from the course dataset.

```bash
python app.py
```

## Switching providers

The app follows the course's provider convention (Gemini default). Override via environment variables — no code change needed:

| Variable | Default | Options |
|---|---|---|
| `PROVIDER` | `gemini` | `gemini`, `openai` |
| `CHAT_MODEL` | `gemini-3.7-flash` | `gpt-5.6-luna`, or any newer model id |
| `EMBED_PROVIDER` | `gemini` | `gemini`, `openai` |
| `EMBED_MODEL` | `gemini-embedding-001` | `text-embedding-3-small` |

Each embedding provider has its own prebuilt store (same corpus, same chunks); the app downloads the one matching `EMBED_PROVIDER`.

## Deploying on Hugging Face Spaces

- The YAML header at the top of this README is the Space configuration. `python_version: "3.12"` is required — Spaces default to Python 3.10, and the `tai-aitutor` install fails there.
- Set `GOOGLE_API_KEY` (and any optional provider keys) as **Secrets** in the Space settings — never commit keys.
- `.github/workflows/main.yml` syncs every push on `main` to the Space (requires an `HF_TOKEN` secret in the GitHub repository settings).
- The free CPU tier is enough: models are called via APIs, and the store download (~100 MB per cold start) fits comfortably in the Space's ephemeral disk.

## Rebuilding the vector store with your own data

`scripts/build_vector_store.py` is the exact script that built the hosted stores. It rebuilds from `ai_tutor_knowledge.jsonl` (auto-downloaded) with either embedding provider — or point `--input` at your own JSONL (one object per document with `doc_id`, `name`, `url`, `source`, `content` fields) to serve your own data:

```bash
python scripts/build_vector_store.py --model gemini-embedding-001 --dimensions 1536 --normalize
```
