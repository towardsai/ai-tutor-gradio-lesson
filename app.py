# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import chromadb
import gradio as gr
import logfire

# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata,QueryEngineTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logfire.configure()

system_message_openai_agent = """You are an AI teacher, answering questions from students of an applied AI course on Large Language Models (LLMs or llm) and Retrieval Augmented Generation (RAG) for LLMs. 
Topics covered include training models, fine-tuning models, giving memory to LLMs, prompting tips, hallucinations and bias, vector databases, transformer architectures, embeddings, RAG frameworks such as 
Langchain and LlamaIndex, making LLMs interact with tools, AI agents, reinforcement learning with human feedback (RLHF). Questions should be understood in this context. Your answers are aimed to teach 
students, so they should be complete, clear, and easy to understand. Use the available tools to gather insights pertinent to the field of AI. To answer student questions, always use the all_sources_info 
tool plus another one simultaneously. Decompose the user question into TWO sub questions (you are limited to two sub-questions) one for each tool. Meaning that should be using two tools in total for each user question.

These are the guidelines to consider if you decide to create sub questions:
* Be as specific as possible
* The two sub questions should be relevant to the user question
* The two sub questions should be answerable by the tools provided

Only some information returned by the tools might be relevant to the question, so ignore the irrelevant part and answer the question with what you have. Your responses are exclusively based on the output provided 
by the tools. Refrain from incorporating information not directly obtained from the tool's responses. When the conversation deepens or shifts focus within a topic, adapt your input to the tools to reflect these nuances. 
This means if a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Provide 
comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved. 
Should the tools repository lack information on the queried topic, politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation. 
At the end of your answers, always invite the students to ask deeper questions about the topic if they have any. Make sure reformulate the question to the tool to capture this new angle or more profound layer of inquiry. 
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so 
they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.
"""
TEXT_QA_TEMPLATE = """
You must answer only related to AI, ML, Deep Learning and related concept queries. You should not 
answer on your own, Should answer from the retrieved chunks. If the query is not relevant to AI, you don't know the answer.
"""


if not os.path.exists("data/ai_tutor_knowledge"):
    os.makedirs("data/ai_tutor_knowledge")
    # Download the vector database from the Hugging Face Hub if it doesn't exist locally
    # https://huggingface.co/datasets/jaiganesan/ai_tutor_knowledge_vector_Store/tree/main
    logfire.warn(
        f"Vector database does not exist at 'data/ai_tutor_knowledge', downloading from Hugging Face Hub"
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="jaiganesan/ai_tutor_knowledge_vector_Store",
        local_dir="data",
        repo_type="dataset",
    )
    logfire.info(f"Downloaded vector database to 'data/ai_tutor_knowledge'")


def setup_database(db_collection):
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

    cohere_reranker = CohereRerank(top_n=7, model="embed-english-v3.0")

    index_query_engine = index.as_query_engine(
        llm=Settings.llm,
        text_qa_template=TEXT_QA_TEMPLATE,
        streaming=True,
        # node_postprocessors=[cohere_reranker],
    )
    return index_query_engine, vector_retriever


DB_NAME = os.getenv("DB_NAME", "ai_tutor_knowledge")
DB_PATH = os.getenv("DB_PATH", f"scripts/{DB_NAME}")

query_engine, vector_retriever = setup_database(DB_NAME)

# Constants
CONCURRENCY_COUNT = int(os.getenv("CONCURRENCY_COUNT", 64))


__all__ = [
    "CONCURRENCY_COUNT",
]


def update_query_engine_tools(query_engine_, vector_retriever_):

    tools = [
        # QueryEngineTool(
        #     query_engine=query_engine_,
        #     metadata=ToolMetadata(
        #         name="AI_information",
        #         description="""The 'AI_information' tool serves as a comprehensive repository for insights into
        #                     the field of artificial intelligence.""",
        #     ),
        # ),
        RetrieverTool(
            retriever=vector_retriever_,
            metadata=ToolMetadata(
                name="AI_Information_related_resources",
                description="Retriever retrieves AI, ML, DL related information from Vector store collection.",
            ),
        )
    ]
    return tools


def generate_completion(query, history, memory):
    with logfire.span("Running query"):
        logfire.info(f"User query: {query}")

        chat_list = memory.get()

        if len(chat_list) != 0:
            user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
            if len(user_index) > len(history):
                user_index_to_remove = user_index[len(history)]
                chat_list = chat_list[:user_index_to_remove]
                memory.set(chat_list)

        logfire.info(f"chat_history: {len(memory.get())} {memory.get()}")
        logfire.info(f"gradio_history: {len(history)} {history}")

        llm = OpenAI(temperature=1, model="gpt-4o-mini", max_tokens=None)

        client = llm._get_client()
        logfire.instrument_openai(client)

        agent_tools = update_query_engine_tools(query_engine, vector_retriever)

        agent = OpenAIAgent.from_tools(
            llm=Settings.llm,
            memory=memory,
            tools=agent_tools,
            system_prompt=system_message_openai_agent,
        )

    completion = agent.stream_chat(query)

    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str

def vote(data: gr.LikeData):
    pass
def save_completion(completion, history):
    pass

with gr.Blocks(
    fill_height=True,
    title="Towards AI 🤖",
    analytics_enabled=True,
) as demo:

    memory_state = gr.State(
        lambda: ChatSummaryMemoryBuffer.from_defaults(
            token_limit=120000,
        )
    )
    chatbot = gr.Chatbot(
        scale=1,
        placeholder="<strong>Towards AI 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
        show_label=False,
        likeable=True,
        show_copy_button=True,
    )
    chatbot.like(vote, None, None)

    gr.ChatInterface(
        fn=generate_completion,
        chatbot=chatbot,
        additional_inputs=[memory_state],
    )

if __name__ == "__main__":
    Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    demo.queue(default_concurrency_limit=CONCURRENCY_COUNT)
    demo.launch(debug=True, share=True)