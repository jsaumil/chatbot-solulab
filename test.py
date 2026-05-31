from langchain_ollama.chat_models import ChatOllama
llm = ChatOllama(
        base_url="https://25f2-34-125-183-131.ngrok-free.app",
        model="qwen3:8b",
        think = False
    )

llm.invoke("Hello, how are you?")