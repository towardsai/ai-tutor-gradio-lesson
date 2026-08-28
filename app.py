"""AI Tutor — a Gradio chat app over the course knowledge base.

The deployable version of the Tutor Chat built across the course notebooks:
tai-aitutor for retrieval and generation (the Section 9+ toolkit), the
hand-rolled summary memory from "Enabling Conversational Memory" (Section 9),
and native SDK streaming. Runs locally (`python app.py`) and on Hugging Face
Spaces.
"""

# Standard Library Imports
import json
import logging
import os
import pathlib
import shutil
import zipfile

# Third-party Imports
import gradio as gr
import requests

# tai-aitutor — the course toolkit
from tai_aitutor import (
    EMBED_DIM,         # the toolkit's fixed embedding width (1536)
    build_rag_prompt,  # the RAG prompt, visible
    configure,         # provider + model selection (replaces Settings)
    generate,          # one prompt in, one reply out
    get_collection,    # the Chroma collection IS the index
    n_tokens,          # token counting for the memory budget
    route,             # one typed classification call
    search,            # dense retrieval, with scores and metadata
    setup_notebook,    # API keys from .env (or real env vars on a Space)
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)  # hide the per-call "AFC is enabled" info line

# --------------------------------------------------------------------------- #
# Provider selection — the same values as the course setup cells. Override via
# environment variables (e.g. in the Space settings) without touching code.
# --------------------------------------------------------------------------- #
CHAT_MODEL_FOR = {"gemini": "gemini-3.7-flash", "openai": "gpt-5.6-luna"}
EMBED_MODEL_FOR = {"gemini": "gemini-embedding-001", "openai": "text-embedding-3-small"}
KEY_FOR = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}

PROVIDER = os.getenv("PROVIDER", "gemini")              # "gemini" | "openai"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "gemini")  # "gemini" | "openai"
CHAT_MODEL = os.getenv("CHAT_MODEL", CHAT_MODEL_FOR[PROVIDER])
EMBED_MODEL = os.getenv("EMBED_MODEL", EMBED_MODEL_FOR[EMBED_PROVIDER])
REQUIRED_KEYS = sorted({KEY_FOR[PROVIDER], KEY_FOR[EMBED_PROVIDER]})

PROMPT_SYSTEM_MESSAGE = """You are an AI teacher, answering questions from students of an applied AI course on Large Language Models (LLMs or llm) and Retrieval Augmented Generation (RAG) for LLMs.
Topics covered include LLM APIs (OpenAI, Gemini), agent frameworks (LangChain, LangGraph, LlamaIndex, Deep Agents), Claude Code, building and deploying AI agents, MCP servers and tool use, retrieval augmented generation, giving memory to LLMs and agents, prompting tips, embeddings and vector databases, and fine-tuning models. Questions should be understood in this context.
Your answers are aimed to teach students, so they should be complete, clear, and easy to understand.

You must answer only questions related to AI, ML, Deep Learning and related concepts. If the query is not relevant to AI, politely state that you don't know the answer as it's outside your scope.
When context excerpts from the course knowledge base are provided, ground your answer exclusively in them. Refrain from incorporating information not found in the excerpts. Only some excerpts might be relevant to the question, so ignore the irrelevant parts and answer the question with what you have.
Should the excerpts lack information on the queried topic (even if AI-related), politely inform the user that the question transcends the bounds of your current knowledge base.

At the end of your answers, always invite the students to ask deeper questions about the topic if they have any.
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.
"""

# --------------------------------------------------------------------------- #
# The knowledge base: a prebuilt Chroma vector store, downloaded once.
# Same stores, same loading code as the Section 9 notebooks — built by
# scripts/build_vector_store.py from ai_tutor_knowledge.jsonl (788 documents).
# --------------------------------------------------------------------------- #
STORE_BASE = "https://huggingface.co/datasets/towardsai-tutors/full-stack-ai-engineering-data/resolve/main/vector_stores/"
STORE_FOR = {
    "gemini": "ai_tutor_knowledge-gemini-embedding-001-1536d-norm",
    "openai": "ai_tutor_knowledge-text-embedding-3-small-1536d",
}
DATASET_SHA = "0e801a3f"  # ties each store zip to the exact corpus file it was built from
STORES_DIR = pathlib.Path("prebuilt_stores")

_collection = None  # opened once per process by get_knowledge_base()


def download_knowledge_base_if_not_exists() -> pathlib.Path:
    """Download and unzip the prebuilt vector store for EMBED_PROVIDER, once."""
    slug = STORE_FOR[EMBED_PROVIDER]
    store_dir = STORES_DIR / slug
    if store_dir.exists():
        return store_dir

    zip_path = pathlib.Path(f"{slug}-{DATASET_SHA}.zip")
    if not zip_path.exists():
        logging.warning(f"Vector store not found locally, downloading {zip_path.name} (~100 MB, one time)...")
        with requests.get(STORE_BASE + zip_path.name, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)

    STORES_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(STORES_DIR)
    logging.info(f"Vector store ready at '{store_dir}'")
    return store_dir


def get_knowledge_base():
    """Open the prebuilt store as a Chroma collection (the collection IS the index)."""
    global _collection
    if _collection is None:
        store_dir = download_knowledge_base_if_not_exists()
        manifest = json.loads((store_dir / "manifest.json").read_text())
        _collection = get_collection(manifest["collection_name"], path=str(store_dir / "chroma"))
        assert _collection.count() == manifest["counts"]["chunks"], "chunk count ≠ manifest"
        assert manifest["dimensions"]["actual"] == EMBED_DIM, "store width ≠ the toolkit's EMBED_DIM"
        logging.info(
            f"Knowledge base loaded: {_collection.count():,} chunks | "
            f"{manifest['embedding_model']} @ {manifest['dimensions']['actual']}d"
        )
    return _collection


# --------------------------------------------------------------------------- #
# Conversational memory — the hand-rolled summary strategy from the
# "Enabling Conversational Memory" lesson (Section 9), held in gr.State.
# --------------------------------------------------------------------------- #
class TutorMemory:
    """The complete transcript plus a visible resend strategy: keep recent turns
    verbatim and fold older ones into a running summary once they exceed the
    token budget — the job the old ChatSummaryMemoryBuffer did opaquely."""

    def __init__(self, window_turns=2, summarize_after_tokens=120_000):
        self.window_turns = window_turns
        self.summarize_after_tokens = summarize_after_tokens
        self.messages = []  # the COMPLETE transcript — never trimmed
        self._summary = ""  # running summary of folded turns
        self._folded = 0    # how many leading messages the summary already covers

    def context(self):
        """What actually gets resent to the model this turn."""
        live = self.messages[self._folded:]
        live_tokens = sum(n_tokens(m["content"]) for m in live)
        keep = 2 * self.window_turns  # one turn = user + assistant
        if live_tokens > self.summarize_after_tokens and len(live) > keep:
            fold, live = live[:-keep], live[-keep:]
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in fold)
            self._summary = generate(  # the lossy rewrite — a paraphrase replaces the turns
                "Fold the new turns into the running summary. Keep every personal "
                f"fact and preference.\n\nRUNNING SUMMARY:\n{self._summary or '(empty)'}\n\n"
                f"NEW TURNS:\n{transcript}",
                system="You compress conversations without losing facts.",
            )
            self._folded += len(fold)
            live = self.messages[self._folded:]
        context = list(live)
        if self._summary:
            context.insert(0, {"role": "user", "content": f"(Conversation so far, summarized: {self._summary})"})
        return context

    def record(self, question, reply):
        """Evidence went in the request; only the question enters the transcript."""
        self.messages += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": reply},
        ]

    def sync_with(self, history):
        """Keep memory in step with the visible Gradio history — the UI is the
        source of truth. A shorter history (edit, retry, clear) truncates
        memory, exactly as the old app reconciled its buffer; a longer one
        (a conversation restored from the saved-history panel) is adopted as
        the transcript."""
        visible = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        history_turns = sum(1 for m in visible if m["role"] == "user")
        user_indexes = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if len(user_indexes) > history_turns:    # the UI went back: truncate
            cut = user_indexes[history_turns]
            self.messages = self.messages[:cut]
            if self._folded > len(self.messages):  # the summary covered removed turns
                self._summary, self._folded = "", 0
        elif len(user_indexes) < history_turns:  # a restored conversation: adopt it
            self.messages = visible
            self._summary, self._folded = "", 0


# --------------------------------------------------------------------------- #
# One tutor turn: route → (maybe) condense + retrieve → grounded, streamed
# reply — the Tutor Chat composition from the Section 9 memory lesson.
# --------------------------------------------------------------------------- #
RAG_ROUTES = {
    "retrieve": "Questions needing a course-knowledge lookup: AI/ML/LLM concepts, techniques, 'how does X work'.",
    "direct": (
        "Turns answerable from the conversation itself: rewordings, summaries of "
        "what was already said, formatting requests, greetings."
    ),
    "reject": "Anything off-topic for an AI course: cooking, travel, medical advice, politics.",
}

REJECT_MESSAGE = (
    "I'm your AI-course tutor, so I'll pass on that one — it's outside my scope. "
    "Ask me anything about LLMs, RAG, or the other topics of the course!"
)

TOP_K = 15  # retrieved chunks per query (same as the old app)


def stream_reply(messages, system=None):
    """Stream one model call over a message list, yielding text deltas.

    Streaming lives at the SDK level — one native call per provider, exactly as
    the memory lesson's streaming section shows. The message list is flattened
    to a transcript, the provider-neutral pattern from the same lesson.
    """
    transcript = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
    prompt = transcript + "\n\nAssistant:"

    if PROVIDER == "gemini":
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client()
        for chunk in client.models.generate_content_stream(
            model=CHAT_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(system_instruction=system),
        ):
            yield chunk.text or ""

    elif PROVIDER == "openai":
        from openai import OpenAI

        stream = OpenAI().responses.create(
            model=CHAT_MODEL, instructions=system, input=prompt, stream=True
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta

    else:
        # Any other provider is configured through the toolkit
        # (OpenAI-compatible endpoints); fall back to one non-streamed call.
        yield generate(prompt, system=system)


def generate_completion(query, history, memory):
    """One chat turn, called by gr.ChatInterface with the visible history and
    the per-session TutorMemory from gr.State."""
    logging.info(f"User query: {query}")
    if memory is None:  # an example click passes no memory; history sync rebuilds context
        memory = TutorMemory()
    memory.sync_with(history)

    last_exchange = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in memory.messages[-2:])
    decision = route(
        f"Conversation so far:\n{last_exchange or '(start)'}\n\nNew question: {query}",
        routes=RAG_ROUTES,
    )
    logging.info(f"Route: [{decision.route}] {decision.reason}")

    if decision.route == "reject":
        memory.record(query, REJECT_MESSAGE)
        yield REJECT_MESSAGE
        return

    prompt = query
    if decision.route == "retrieve":
        search_query = query
        if memory.messages:  # the condense step: follow-ups become standalone queries
            search_query = generate(
                "Rewrite the follow-up below as ONE standalone search query for a "
                "technical knowledge base. Reply with the query only.\n\n"
                f"Conversation:\n{last_exchange}\n\nFollow-up: {query}"
            ).strip()
            logging.info(f"Condensed query: {search_query!r}")
        hits = search(search_query, get_knowledge_base(), top_k=TOP_K)
        logging.info(f"Retrieved {len(hits)} chunks")
        prompt = build_rag_prompt(query, hits)  # evidence goes in the REQUEST...

    context = memory.context() + [{"role": "user", "content": prompt}]
    answer = ""
    for delta in stream_reply(context, system=PROMPT_SYSTEM_MESSAGE):
        answer += delta
        yield answer
    memory.record(query, answer)  # ...and only the question goes in the transcript


# Starter questions, each grounded in the current corpus (verified against its sources)
EXAMPLE_QUESTIONS = [
    "How do I build an agent with LangGraph, and how does it keep state?",
    "What is an MCP server, and how do I connect one?",
    "How do I give an AI agent memory across a long conversation?",
    "What's new in LangChain v1?",
]


def launch_ui():
    with gr.Blocks(
        fill_height=True,
        title="AI Tutor 🤖",
        analytics_enabled=False,
    ) as demo:

        memory_state = gr.State(TutorMemory)  # one TutorMemory per session

        gr.Markdown(
            "# 🤖 AI Tutor\n"
            "Ask anything about LLM APIs, agent frameworks, Claude Code, RAG, or "
            "agent engineering — answers are grounded in the course knowledge base "
            "(788 documents) and stream in as they are generated."
        )

        chatbot = gr.Chatbot(
            scale=1,
            placeholder=(
                "<strong>Welcome! 👋</strong><br>"
                "I answer with excerpts retrieved from the course corpus: LangChain, "
                "LangGraph, LlamaIndex, OpenAI, Claude Code and Deep Agents "
                "documentation, plus the course's agent-engineering lessons.<br>"
                "Pick an example below, or ask your own question."
            ),
            show_label=False,
            buttons=["copy"],
        )

        gr.ChatInterface(
            fn=generate_completion,
            chatbot=chatbot,
            additional_inputs=[memory_state],
            examples=[[q, None] for q in EXAMPLE_QUESTIONS],
            cache_examples=False,  # run starters live on click — Spaces otherwise pre-runs them at startup
            save_history=True,  # previous conversations, kept in the browser
        )

    demo.queue(default_concurrency_limit=64)
    demo.launch(
        theme=gr.themes.Default(primary_hue="indigo"),  # Gradio 6: theme is set at launch
        debug=False,
        share=False,  # set share=True for a temporary public link
    )


if __name__ == "__main__":
    # Load API keys (.env locally; real env vars on a Space) and fail fast if missing
    setup_notebook(required_keys=REQUIRED_KEYS)

    # Select provider and models for the toolkit (replaces the old global Settings)
    cfg = configure(
        provider=PROVIDER,
        chat_model=CHAT_MODEL,
        embed_provider=EMBED_PROVIDER,
        embed_model=EMBED_MODEL,
    )
    logging.info(f"Configured: {cfg}")

    # Download the knowledge base if it doesn't exist and open it
    get_knowledge_base()

    # Launch the UI
    launch_ui()
