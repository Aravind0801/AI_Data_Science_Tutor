import streamlit as st
import os
import uuid 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API key is missing! Please set it as an environment variable.")

system_prompt = """
You are an expert Data Science Tutor. Your role is to assist users with data science-related questions only.
Provide clear, detailed, and accurate explanations, including examples where necessary.
If a user asks about a topic unrelated to data science, politely decline.
"""

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)

def get_session_history(session_id: str):
    return ChatMessageHistory()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())  # Unique session ID for each user

conversation = RunnableWithMessageHistory(llm, get_session_history=get_session_history)

st.title("🤖 AI Conversational Data Science Tutor 📊")
st.markdown("Ask any **data science-related** questions! (ML, AI, Stats, Python, etc.)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.button("🗑️ Clear Chat"):
    st.session_state["messages"] = []
    st.rerun()

for chat in st.session_state["messages"]:
    role = "👤 **You:**" if chat["is_user"] else "🤖 **Tutor:**"
    st.markdown(f"{role} {chat['content']}")

user_input = st.text_input("Your question:")

if user_input:

    response = conversation.invoke(
        {"input": f"{system_prompt}\nUser: {user_input}"},
        config={"configurable": {"session_id": st.session_state["session_id"]}}
    )


    ai_response = response.content

    
    st.session_state["messages"].append({"content": user_input, "is_user": True})
    st.session_state["messages"].append({"content": ai_response, "is_user": False})

    
    st.markdown(f"👤 **You:** {user_input}")
    st.markdown(f"🤖 **Tutor:** {ai_response}")


with st.expander("Full Chat History"):
    for chat in st.session_state["messages"]:
        role = "👤 **You:**" if chat["is_user"] else "🤖 **Tutor:**"
        st.write(f"{role} {chat['content']}")
