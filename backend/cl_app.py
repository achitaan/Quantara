import os
from dotenv import load_dotenv

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

# --- LangChain Workflow Setup ---
workflow = StateGraph(state_schema=MessagesState)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
langgraph_app = workflow.compile(checkpointer=memory)

# --- Authentication ---
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Replace with your own user lookup logic
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    return None


# --- Chat Resume (optional) ---
@cl.on_chat_resume
async def on_chat_resume(thread):
    # You can restore memory or state here if needed
    pass


# --- Chat Start (optional) ---
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Connected to Chainlit with LangChain memory!").send()


# --- Message Handler ---
@cl.on_message
async def main(message: cl.Message):
    answer = cl.Message(content="")
    await answer.send()

    config: RunnableConfig = {
        "configurable": {"thread_id": cl.context.session.thread_id}
    }

    # Stream response from LangChain agent
    for msg, _ in langgraph_app.stream(
        {"messages": [HumanMessage(content=message.content)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, AIMessageChunk):
            answer.content += msg.content  # type: ignore
            await answer.update()
