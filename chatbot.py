from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64 as b64
from PIL import Image
from io import BytesIO
import os
import json
import requests
import uuid

load_dotenv()
DB_URL = os.getenv("DB_URL")
IMAGE_GEN_URL = os.getenv("IMAGE_GEN_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatOllama(
        base_url="https://25f2-34-125-183-131.ngrok-free.app",
        model="qwen3:8b",
        think = False
    )
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store setup
with open("data.json", "r") as f:
    verses = json.load(f)

vector_store = FAISS.from_texts(
    texts=[v["text"] for v in verses],
    embedding=embedding
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Tools
@tool
def rag_tool(query: str):
    """
    RAG Tool
    """
    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {"query": query, "context": context, "metadata": metadata}

@tool
def image_generation_tool(prompt: str):
    """
    Image Generation Tool
    """
    try:
        safe_prompt = f"Sacred Christian artwork, reverent, biblical, classical style: {prompt}"
        response = requests.post(
            IMAGE_GEN_URL, 
            json={"prompt": safe_prompt}, 
            )
        response.raise_for_status()
        image_b64 = response.json().get("image", "")
        # Save to temp file
        save_path = os.path.join(os.getcwd(), f"logos_{uuid.uuid4().hex[:8]}.png")
        if image_b64:
            img = Image.open(BytesIO(b64.b64decode(image_b64)))
            img.save(save_path)
        return {"status": "success", "path": save_path, "prompt_used": safe_prompt}
    except Exception as e:
        return {"status": "error", "error": str(e)}

tools = [rag_tool, image_generation_tool]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def main(question, thread_id=None):
    def chat_node(state: ChatState):
        """LLM node that may answer or request a tool call"""
        system = SystemMessage(content="""You are Logos, a Christianity-focused AI assistant. You follow these rules absolutely:
 
        SCRIPTURE INTEGRITY:
        - NEVER fabricate, paraphrase, or invent Bible verse references or text.
        - When referencing a verse, ALWAYS use rag_tool first to retrieve the actual text.
        - If a verse cannot be found, explicitly say "I could not verify this reference" — never guess.
        
        CONTENT SAFETY:
        - Refuse requests to rewrite scripture to support harmful ideologies.
        - Do not put fabricated words or quotes in the mouth of Jesus, God, or the Holy Spirit.
        - Decline respectfully but firmly if asked to produce heretical, hateful, or offensive content.
        
        TONE:
        - Be warm, respectful, and scholarly. Balance pastoral care with theological accuracy.
        - For difficult questions (e.g. theodicy, doubt), respond with empathy and nuance.
        
        DENOMINATION:
        - When a tradition is specified, use appropriate terminology (e.g. "Eucharist" for Catholic, "Lord's Supper" for Protestant).
        - For contested theological points, note where traditions differ rather than asserting one view as universal.
        
        IMAGE GENERATION:
        - Use image_generation_tool only when the user explicitly wants artwork or an image.
        - Ensure image prompts are reverent and appropriate.""")
        response = llm_with_tools.invoke([system] + state['messages'])
        return {"messages":[response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
        checkpointer.setup()

        chatbot = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
        messages = [HumanMessage(content=question)]
        result = chatbot.invoke({"messages": messages}, config=config)

    return result


if __name__ == "__main__":
    chatbot = main("make a painting of Jesus teaching his disciples")
    print(chatbot)