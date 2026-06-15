import os
from dotenv import load_dotenv
# Load environment variables from .env file immediately at startup
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import FakeListLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from controllers import file_router, proposal_router

app = FastAPI()

# Register routers
app.include_router(file_router)
app.include_router(proposal_router)

@app.get("/")
def checkHealth():
    return {
        "message " : "Hello World !",
        "status" : "ok"
    }

@app.get("/test-langchain")
def run_langchain_test():
    # 1. Create a simple prompt template
    prompt = PromptTemplate.from_template("Tell me a short joke about {topic}")

    # 2. Set up a fake LLM that returns a predefined response
    responses = ["Why do programmers wear glasses? Because they can't C#!"]
    llm = FakeListLLM(responses=responses)

    # 3. Chain them together
    chain = prompt | llm

    # 4. Run the chain
    try:
        topic = "programming"
        result = chain.invoke({"topic": topic})
        return {
            "status": "success",
            "prompt": f"Tell me a short joke about {topic}",
            "response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# =====================================================================
# ADDED: Real LangChain Configuration with Gemini and Memory Window
# =====================================================================

# 1. Initialize the live Gemini model (automatically checks for GOOGLE_API_KEY from .env)
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7,
    max_retries=5
)

# 2. Set up clean in-memory state tracking for dynamic user sessions
sessions_db = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in sessions_db:
        sessions_db[session_id] = InMemoryChatMessageHistory()
    return sessions_db[session_id]

# 3. Chat prompt template layout mapping chat history injection
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, highly concise AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 4. Create the final runnable pipeline managed by automated message histories
runnable_chain = chat_prompt | llm

chain_with_history = RunnableWithMessageHistory(
    runnable_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


# 5. Define Pydantic schema for structured request handling
class ChatRequest(BaseModel):
    user_input: str
    session_id: str = "default_user"


# 6. Exposed Production Endpoint for chat execution
@app.post("/chat")
def run_real_chat(request: ChatRequest):
    try:
        # Pull history state window automatically & trim context under the hood
        history = get_session_history(request.session_id)
        
        # Keep memory window small: drop oldest messages if context gets too long (k=3 window logic)
        # Each turn has 2 messages (Human + AI), so 6 messages maximum in history
        if len(history.messages) > 6:
            history.messages = history.messages[-6:]

        # Execute conversation stream
        result = chain_with_history.invoke(
            {"input": request.user_input},
            config={"configurable": {"session_id": request.session_id}}
        )

        return {
            "status": "success",
            "session_id": request.session_id,
            "response": result.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))