# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import gradio as gr
from huggingface_hub import snapshot_download

# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_SYSTEM_MESSAGE = """You are an AI teacher, answering questions from students of an applied AI course on Large Language Models (LLMs or llm) and Retrieval Augmented Generation (RAG) for LLMs.
Topics covered include training models, fine-tuning models, giving memory to LLMs, prompting tips, hallucinations and bias, vector databases, transformer architectures, embeddings, RAG frameworks such as 
Langchain and LlamaIndex, making LLMs interact with tools, AI agents, reinforcement learning with human feedback (RLHF). Questions should be understood in this context.
Your answers are aimed to teach students, so they should be complete, clear, and easy to understand. Use the available tools to gather insights pertinent to the field of AI.

To find relevant information for answering student questions, always use the "AI_Information_related_resources" tool.

You must answer only related to AI, ML, Deep Learning and related concepts queries. If the query is not relevant to AI, politely state that you don't know the answer as it's outside your scope.
Always leverage the retrieved documents (information returned by the tool) to answer the questions. Your responses are exclusively based on the output provided by the tools. Refrain from incorporating information not directly obtained from the tool's responses. Only some information returned by the tool might be relevant to the question, so ignore the irrelevant part and answer the question with what you have.

If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Provide 
comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved. 
Should the tool response lack information on the queried topic (even if AI-related), politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation. 

At the end of your answers, always invite the students to ask deeper questions about the topic if they have any.
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so 
they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.
"""


def download_knowledge_base_if_not_exists():
    """Download the knowledge base from the Hugging Face Hub if it doesn't exist locally"""
    if not os.path.exists("data/ai_tutor_knowledge"):
        os.makedirs("data/ai_tutor_knowledge")

        logging.warning(
            f"Vector database does not exist at 'data/', downloading from Hugging Face Hub..."
        )
        snapshot_download(
            repo_id="jaiganesan/ai_tutor_knowledge_vector_Store",
            local_dir="data/ai_tutor_knowledge",
            repo_type="dataset",
        )
        logging.info(f"Downloaded vector database to 'data/ai_tutor_knowledge'")


def get_tools(db_collection="ai_tutor_knowledge"):
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=Settings.embed_model,
        use_async=True,
    )
    tools = [
        RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="AI_Information_related_resources",
                description="Useful for info related to artificial intelligence, ML, deep learning. It gathers the info from local data.",
            ),
        )
    ]
    return tools


def generate_completion(query, history, memory):
    logging.info(f"User query: {query}")

    # Manage memory
    chat_list = memory.get()
    if len(chat_list) != 0:
        history_turns = sum(1 for msg in history if msg["role"] == "user")
        user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
        if len(user_index) > history_turns:
            user_index_to_remove = user_index[history_turns]
            chat_list = chat_list[:user_index_to_remove]
            memory.set(chat_list)
    logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
    logging.info(f"gradio_history: {len(history)} {history}")

    # Create agent
    tools = get_tools(db_collection="ai_tutor_knowledge")
    agent = OpenAIAgent.from_tools(
        llm=Settings.llm,
        memory=memory,
        tools=tools,
        system_prompt=PROMPT_SYSTEM_MESSAGE,
    )

    # Generate answer
    completion = agent.stream_chat(query)
    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str


def launch_ui():
    with gr.Blocks(
        fill_height=True,
        title="AI Tutor 🤖",
        analytics_enabled=True,
    ) as demo:

        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        chatbot = gr.Chatbot(
            type="messages",
            scale=1,
            placeholder="<strong>AI Tutor 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
            show_label=False,
            show_copy_button=True,
        )

        gr.ChatInterface(
            fn=generate_completion,
            chatbot=chatbot,
            additional_inputs=[memory_state],
            type="messages",
        )

        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, share=True) # Set share=True to share the app online


if __name__ == "__main__":
    # Download the knowledge base if it doesn't exist
    download_knowledge_base_if_not_exists()

    # Set up llm and embedding model
    Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # launch the UI
    launch_ui()