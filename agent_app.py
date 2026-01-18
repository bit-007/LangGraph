from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Dict, Any, Optional
from langgraph.graph import add_messages
from tools import ask_user, get_policy_details, get_claim_status
from llm_utils import run_llm, client   # ✅ CORRECT
from prompts import *
from env_loader import tracer
import re, json
from tracing_utils import trace_agent
import logging
from tools import (
    ask_user,
    get_policy_details,
    get_claim_status,
    get_billing_info,
    get_payment_history,
    get_auto_policy_details,
)

logger = logging.getLogger(__name__)






class GraphState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_input: str
    conversation_history: Optional[str]
    n_iteration: Optional[int]
    customer_id: Optional[str]
    policy_number: Optional[str]
    claim_id: Optional[str]
    next_agent: Optional[str]
    task: Optional[str]
    justification: Optional[str]
    end_conversation: Optional[bool]
    requires_human_escalation: bool
    final_answer: Optional[str]

    # 🔹 REQUIRED FOR STREAMLIT PAUSE
    needs_user_input: Optional[bool]
    question: Optional[str]
    missing_info: Optional[str]
    needs_clarification: Optional[bool]
    clarification_question: Optional[str]
    user_clarification: Optional[str]



@trace_agent
def supervisor_agent(state):
    print("---SUPERVISOR AGENT---")

    # ---------------------------
    # PHASE 1: Handle clarification return from Streamlit
    # ---------------------------
    if state.get("needs_clarification"):
        print("✅ Processing user clarification")

        user_clarification = state.get("user_clarification", "")
        clarification_question = state.get("clarification_question", "")

        # Extract IDs from user answer
        policy_match = re.search(r'POL\d{6}', user_clarification)
        if policy_match:
            state["policy_number"] = policy_match.group()
            print(f"📋 Extracted policy number: {state['policy_number']}")

        customer_match = re.search(r'CUST\d{5}', user_clarification)
        if customer_match:
            state["customer_id"] = customer_match.group()
            print(f"👤 Extracted customer ID: {state['customer_id']}")

        claim_match = re.search(r'CLM\d{6}', user_clarification)
        if claim_match:
            state["claim_id"] = claim_match.group()
            print(f"📄 Extracted claim ID: {state['claim_id']}")

        # Update conversation history (THIS is where your updated_history logic lives)
        updated_conversation = (
            state.get("conversation_history", "")
            + f"\nAssistant: {clarification_question}\nUser: {user_clarification}"
        )

        state["conversation_history"] = updated_conversation
        state["needs_clarification"] = False

        # Clean up temporary fields
        state.pop("clarification_question", None)
        state.pop("user_clarification", None)

    # ---------------------------
    # PHASE 2: Extract from history (safety net)
    # ---------------------------
    conversation_history = state.get("conversation_history", "")

    if not state.get("policy_number"):
        m = re.search(r'POL\d{6}', conversation_history)
        if m:
            state["policy_number"] = m.group()
            print(f"📋 Extracted policy number from history: {state['policy_number']}")

    if not state.get("customer_id"):
        m = re.search(r'CUST\d{5}', conversation_history)
        if m:
            state["customer_id"] = m.group()
            print(f"👤 Extracted customer ID from history: {state['customer_id']}")

    if not state.get("claim_id"):
        m = re.search(r'CLM\d{6}', conversation_history)
        if m:
            state["claim_id"] = m.group()
            print(f"📄 Extracted claim ID from history: {state['claim_id']}")

    # ---------------------------
    # PHASE 3: Iteration guard
    # ---------------------------
    n_iter = state.get("n_iteration", 0) + 1
    state["n_iteration"] = n_iter
    print(f"🔢 Supervisor iteration: {n_iter}")

    if n_iter >= 5:
        print("⚠️ Max iterations reached — escalating to human")
        state["requires_human_escalation"] = True
        state["next_agent"] = "human_escalation_agent"
        return state

    # ---------------------------
    # PHASE 4: If specialist previously asked for POLICY NUMBER → ask user ONCE
    # ---------------------------
    history_lower = conversation_history.lower()

    if any(phrase in history_lower for phrase in [
        "please provide your policy number",
        "provide your policy number",
        "could you provide your policy number"
    ]) and not state.get("policy_number"):

        print("🚨 Need policy number → asking user")

        ask = ask_user("What is your policy number?", "policy number")

        return {
            "needs_user_input": True,
            "question": ask["question"],
            "missing_info": ask.get("missing_info", ""),
            "conversation_history": conversation_history,
            "n_iteration": n_iter,
            "policy_number": state.get("policy_number", ""),
            "customer_id": state.get("customer_id", "")
        }

    # ---------------------------
    # PHASE 5: If specialist asked for CLAIM ID → ask user ONCE
    # ---------------------------
    if any(phrase in history_lower for phrase in [
        "please provide your claim id",
        "provide your claim id",
        "could you provide your claim id"
    ]) and not state.get("claim_id"):

        print("🚨 Need claim ID → asking user")

        ask = ask_user("What is your claim ID?", "claim ID")

        return {
            "needs_user_input": True,
            "question": ask["question"],
            "missing_info": ask.get("missing_info", ""),
            "conversation_history": conversation_history,
            "n_iteration": n_iter,
            "policy_number": state.get("policy_number", ""),
            "customer_id": state.get("customer_id", ""),
            "claim_id": state.get("claim_id", "")
        }

    # ---------------------------
    # PHASE 6: If we already HAVE a policy number → route directly
    # ---------------------------
    if state.get("policy_number"):
        print("✅ Have policy number → routing to billing_agent")

        updated_conversation = conversation_history + \
            f"\nAssistant: Routing to billing_agent with policy: {state['policy_number']}"

        return {
            "next_agent": "billing_agent",
            "task": "Retrieve premium information for the user's auto insurance policy",
            "justification": "User has provided policy number",
            "conversation_history": updated_conversation,
            "n_iteration": n_iter,
            "policy_number": state.get("policy_number", ""),
            "customer_id": state.get("customer_id", "")
        }

    # ---------------------------
    # PHASE 7: Otherwise → let LLM decide routing
    # ---------------------------
    print("🤖 Calling LLM for routing decision...")

    prompt = SUPERVISOR_PROMPT.format(
        conversation_history=f"Full Conversation:\n{conversation_history}"
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask for missing info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "missing_info": {"type": "string"}
                    },
                    "required": ["question", "missing_info"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # If LLM wants to ask user
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            if tc.function.name == "ask_user":
                args = json.loads(tc.function.arguments)
                ask = ask_user(args["question"], args["missing_info"])

                return {
                    "needs_user_input": True,
                    "question": ask["question"],
                    "missing_info": ask.get("missing_info", ""),
                    "conversation_history": conversation_history,
                    "n_iteration": n_iter,
                    "policy_number": state.get("policy_number", ""),
                    "customer_id": state.get("customer_id", "")
                }

    # Otherwise parse routing decision
    try:
        parsed = json.loads(message.content)
    except:
        parsed = {}

    return {
        "next_agent": parsed.get("next_agent", "general_help_agent"),
        "task": parsed.get("task", "Assist the user with their query."),
        "justification": parsed.get("justification", ""),
        "conversation_history": conversation_history + 
            f"\nAssistant: Routing to {parsed.get('next_agent', 'general_help_agent')}",
        "n_iteration": n_iter,
        "policy_number": state.get("policy_number", ""),
        "customer_id": state.get("customer_id", "")
    }


@trace_agent
def claims_agent_node(state):
    logger.info("🏥 Claims agent started")
    logger.debug(f"Claims agent state: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = CLAIMS_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        claim_id=state.get("claim_id", "Not provided"),
        conversation_history=state.get("conversation_history", "")
    )

    tools = [
        {"type": "function", "function": {
            "name": "get_claim_status",
            "description": "Retrieve claim details",
            "parameters": {"type": "object", "properties": {"claim_id": {"type": "string"}, "policy_number": {"type": "string"}}}
        }}
    ]

    result = run_llm(prompt, tools, {"get_claim_status": get_claim_status})
    
    logger.info("✅ Claims agent completed")
    return {"messages": [("assistant", result)]}

@trace_agent
def final_answer_agent(state):
    """Generate a clean final summary before ending the conversation"""
    print("---FINAL ANSWER AGENT---")
    logger.info("🎯 Final answer agent started")
    
    user_query = state["user_input"]
    conversation_history = state.get("conversation_history", "")
    
    # Extract the most recent specialist response
    recent_responses = []
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, 'content') and "clarification" not in msg.content.lower():
            recent_responses.append(msg.content)
            if len(recent_responses) >= 2:  # Get last 2 non-clarification responses
                break
    
    specialist_response = recent_responses[0] if recent_responses else "No response available"
    
    prompt = FINAL_ANSWER_PROMPT.format(

        specialist_response=specialist_response,  
        user_query=user_query,
    )
    
    print("🤖 Generating final summary...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    
    final_answer = response.choices[0].message.content
    
    print(f"✅ Final answer: {final_answer}")
    
    # Replace all previous messages with just the final answer
    clean_messages = [("assistant", final_answer)]

    state["final_answer"] = final_answer
    state["end_conversation"] = True
    state["conversation_history"] = conversation_history + f"\nAssistant: {final_answer}"
    state["messages"] = clean_messages
    
    return state


    
@trace_agent
def policy_agent_node(state):
    print("---POLICY AGENT---")
    logger.info("📄 Policy agent started")
    logger.debug(f"Policy agent state: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = POLICY_AGENT_PROMPT.format(
        task=state.get("task"),
        policy_number=state.get("policy_number", "Not provided"),
        customer_id=state.get("customer_id", "Not provided"),
        conversation_history=state.get("conversation_history", "")
    )

    tools = [
        {"type": "function", "function": {
            "name": "get_policy_details",
            "description": "Fetch policy info by policy number",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}}}
        }},
        {"type": "function", "function": {
            "name": "get_auto_policy_details",
            "description": "Get auto policy details",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}}}
        }}
    ]

    print("🔄 Processing policy request...")
    result = run_llm(prompt, tools, {
        "get_policy_details": get_policy_details,
        "get_auto_policy_details": get_auto_policy_details
    })
    
    print("✅ Policy agent completed")
    return {"messages": [("assistant", result)]}

@trace_agent
def billing_agent_node(state):
    print("---BILLING AGENT---")
    print("TASK: ", state.get("task"))
    print("USER QUERY: ", state.get("user_input"))
    print("CONVERSATION HISTORY: ", state.get("conversation_history", ""))
    
    
    prompt = BILLING_AGENT_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", "")
    )

    tools = [
        {"type": "function", "function": {
            "name": "get_billing_info",
            "description": "Retrieve billing information",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}, "customer_id": {"type": "string"}}}
        }},
        {"type": "function", "function": {
            "name": "get_payment_history",
            "description": "Fetch recent payment history",
            "parameters": {"type": "object", "properties": {"policy_number": {"type": "string"}}}
        }}
    ]

    print("🔄 Processing billing request...")
    result = run_llm(prompt, tools, {
        "get_billing_info": get_billing_info,
        "get_payment_history": get_payment_history
    })
    
    print("✅ Billing agent completed")
    
    # Extract and preserve policy number if mentioned in the conversation
    updated_state = {"messages": [("assistant", result)]}
    
    # If we have a policy number in state, preserve it
    if state.get("policy_number"):
        updated_state["policy_number"] = state["policy_number"]
    if state.get("customer_id"):
        updated_state["customer_id"] = state["customer_id"]
        
    # Update conversation history
    current_history = state.get("conversation_history", "")
    updated_state["conversation_history"] = current_history + f"\nBilling Agent: {result}"
    updated_state["end_conversation"] = True   # <-- CRITICAL FIX
    updated_state["next_agent"] = "final_answer_agent"


    
    return updated_state


@trace_agent
def general_help_agent_node(state):
    print("---GENERAL HELP AGENT---")

    user_query = state.get("user_input", "")
    conversation_history = state.get("conversation_history", "")
    task = state.get("task", "General insurance support")

    # Step 1: Retrieve relevant FAQs from the vector DB
    print("🔍 Retrieving FAQs...")
    logger.info("🔍 Retrieving FAQs from vector database")
    results = collection.query(
        query_texts=[user_query],
        n_results=3,
        include=["metadatas", "documents", "distances"]
    )

    # Step 2: Format retrieved FAQs
    faq_context = ""
    if results and results.get("metadatas") and results["metadatas"][0]:
        print(f"📚 Found {len(results['metadatas'][0])} relevant FAQs")
        for i, meta in enumerate(results["metadatas"][0]):
            q = meta.get("question", "")
            a = meta.get("answer", "")
            score = results["distances"][0][i]
            faq_context += f"FAQ {i+1} (score: {score:.3f})\nQ: {q}\nA: {a}\n\n"
    else:
        print("❌ No relevant FAQs found")
        faq_context = "No relevant FAQs were found."

    # Step 3: Format the final prompt
    prompt = GENERAL_HELP_PROMPT.format(
        task=task,
        conversation_history=conversation_history,
        faq_context=faq_context
    )

    print("🤖 Calling LLM for general response...")
    final_answer = run_llm(prompt)

    
    
    print("✅ General help agent completed")
    updated_state = {
                        "messages": [("assistant", final_answer)],
                        "retrieved_faqs": results.get("metadatas", []),
                    }


    updated_state["conversation_history"] = conversation_history + f"\nGeneral Help Agent: {final_answer}"

    return updated_state

@trace_agent
def human_escalation_node(state):
    print("---HUMAN ESCALATION AGENT---")
    logger.warning(f"Escalation triggered - State: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = HUMAN_ESCALATION_PROMPT.format(
        task=state.get("task"),
        #user_query=state.get("user_input"),
        conversation_history=state.get("conversation_history", "")
    )

    print("🤖 Generating escalation response...")
    response = client.chat.completions.create(
    model="gpt-4o-mini",   # ✅ VALID MODEL
    messages=[{"role": "system", "content": prompt}]
    )



    print("🚨 Conversation escalated to human")
    return {
        "final_answer": response.choices[0].message.content,
        "requires_human_escalation": True,
        "escalation_reason": "Customer requested human assistance.",
        "messages": [("assistant", response.choices[0].message.content)]
    }


def decide_next_agent(state):
    # 1) If we need to ask the user → loop back to supervisor
    if state.get("needs_user_input"):
        return "supervisor_agent"

    # 2) If billing or any agent marked the conversation done → go to final_answer_agent
    if state.get("end_conversation"):
        return "final_answer_agent"

    # 3) Otherwise follow next_agent
    return state.get("next_agent", "supervisor_agent")









# ---- Build workflow ----
workflow = StateGraph(GraphState)

workflow.add_node("supervisor_agent", supervisor_agent)
workflow.add_node("policy_agent", policy_agent_node)
workflow.add_node("billing_agent", billing_agent_node)
workflow.add_node("claims_agent", claims_agent_node)
workflow.add_node("general_help_agent", general_help_agent_node)
workflow.add_node("human_escalation_agent", human_escalation_node)
workflow.add_node("final_answer_agent", final_answer_agent)

workflow.set_entry_point("supervisor_agent")

workflow.add_conditional_edges(
    "supervisor_agent",
    decide_next_agent,
    {
        "supervisor_agent": "supervisor_agent",
        "policy_agent": "policy_agent",
        "billing_agent": "billing_agent", 
        "claims_agent": "claims_agent",
        "human_escalation_agent": "human_escalation_agent",
        "general_help_agent": "general_help_agent",
        "final_answer_agent": "final_answer_agent"
    }
)

for node in ["policy_agent", "billing_agent", "claims_agent", "general_help_agent"]:
    workflow.add_edge(node, "supervisor_agent")

workflow.add_edge("final_answer_agent", END)
workflow.add_edge("human_escalation_agent", END)

app = workflow.compile()
