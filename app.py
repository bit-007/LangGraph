import streamlit as st
from agent_app import app   # your compiled LangGraph app

st.set_page_config(page_title="AI Insurance Support", page_icon="🏦")
st.title("🏦 AI Insurance Support Assistant")

# Initialize session state
if "state" not in st.session_state:
    st.session_state["state"] = None
    st.session_state["graph_completed"] = False

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = ""

if "awaiting_clarification" not in st.session_state:
    st.session_state["awaiting_clarification"] = False

if "last_question" not in st.session_state:
    st.session_state["last_question"] = None

# Display chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for role, msg in st.session_state["chat_history"]:
    with st.chat_message(role):
        st.write(msg)

# User input box
user_query = st.chat_input("Ask about your insurance...")

def init_state(user_query: str):
    """Create fresh initial state for a new conversation"""
    return {
        "n_iteration": 0,  # Fixed typo: was "n_iteraton"
        "messages": [],
        "user_input": user_query,
        "user_intent": "",
        "claim_id": "",
        "next_agent": "supervisor_agent",
        "extracted_entities": {},
        "database_lookup_result": {},
        "requires_human_escalation": False,
        "escalation_reason": "",
        "billing_amount": None,
        "payment_method": None,
        "billing_frequency": None,
        "invoice_date": None,
        "conversation_history": f"User: {user_query}",
        "task": "Help user with their query",
        "final_answer": ""
    }

if user_query:
    # Show user message in chat
    st.session_state["chat_history"].append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # Case 1: This is a NEW question (not a clarification)
    if not st.session_state["awaiting_clarification"]:
        # Start fresh graph run
        st.session_state["state"] = init_state(user_query)
        st.session_state["graph_completed"] = False

    else:
        # Case 2: This is the USER'S ANSWER to a clarification
        st.session_state["state"] = {
            **st.session_state["state"],
            "needs_clarification": True,
            "clarification_question": st.session_state["last_question"],
            "user_clarification": user_query,
            "user_input": user_query,
        }
        st.session_state["awaiting_clarification"] = False
        st.session_state["last_question"] = None

    # Run the graph ONLY if not already completed
    if not st.session_state.get("graph_completed", False):
        try:
            state = app.invoke(st.session_state["state"])
            st.session_state["state"] = state
            
            # Mark as completed to prevent re-invocation
            st.session_state["graph_completed"] = True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state["graph_completed"] = True

    state = st.session_state["state"]

    # --- CASE A: Agent needs more info ---
    if state.get("needs_user_input"):
        question = state["question"]

        st.session_state["awaiting_clarification"] = True
        st.session_state["last_question"] = question
        st.session_state["graph_completed"] = False  # Allow next run

        # Show assistant question in chat
        st.session_state["chat_history"].append(("assistant", question))
        with st.chat_message("assistant"):
            st.write(question)

        st.stop()  # wait for user's next message

    # --- CASE B: Conversation is done ---
    final_answer = state.get("final_answer")

    if final_answer:
        st.session_state["chat_history"].append(("assistant", final_answer))
        with st.chat_message("assistant"):
            st.write(final_answer)
    else:
        fallback = "I'm still processing your request. Please continue."
        st.session_state["chat_history"].append(("assistant", fallback))
        with st.chat_message("assistant"):
            st.write(fallback)