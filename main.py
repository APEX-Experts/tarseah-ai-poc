from fastapi import FastAPI
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

app = FastAPI()

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



