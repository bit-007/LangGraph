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
    # PHASE 1: Handle clarification return
    # ---------------------------
    if state.get("needs_clarification"):
        print("✅ Processing user clarification")
        user_clarification = state.get("user_clarification", "")
        clarification_question = state.get("clarification_question", "")

        # Extract IDs
        policy_match = re.search(r'POL\d{6}', user_clarification)
        if policy_match:
            state["policy_number"] = policy_match.group()

        customer_match = re.search(r'CUST\d{5}', user_clarification)
        if customer_match:
            state["customer_id"] = customer_match.group()

        claim_match = re.search(r'CLM\d{6}', user_clarification)
        if claim_match:
            state["claim_id"] = claim_match.group()

        updated_conversation = (
            state.get("conversation_history", "")
            + f"\nAssistant: {clarification_question}\nUser: {user_clarification}"
        )

        state["conversation_history"] = updated_conversation
        state["needs_clarification"] = False
        state.pop("clarification_question", None)
        state.pop("user_clarification", None)

    # ---------------------------
    # PHASE 2: Extract from history
    # ---------------------------
    conversation_history = state.get("conversation_history", "")

    if not state.get("policy_number"):
        m = re.search(r'POL\d{6}', conversation_history)
        if m:
            state["policy_number"] = m.group()

    if not state.get("customer_id"):
        m = re.search(r'CUST\d{5}', conversation_history)
        if m:
            state["customer_id"] = m.group()

    if not state.get("claim_id"):
        m = re.search(r'CLM\d{6}', conversation_history)
        if m:
            state["claim_id"] = m.group()

    # ---------------------------
    # PHASE 3: Iteration guard
    # ---------------------------
    n_iter = state.get("n_iteration", 0) + 1
    state["n_iteration"] = n_iter
    print(f"🔢 Supervisor iteration: {n_iter}")

    if n_iter >= 5:
        print("⚠️ Max iterations → escalating to human")
        # ⚠️ CRITICAL: RETURN immediately with escalation routing
        return {
            "requires_human_escalation": True,
            "next_agent": "human_escalation_agent",
            "task": "Escalate to human support",
            "justification": "Maximum iterations reached without resolution",
            "conversation_history": conversation_history,
            "n_iteration": n_iter
        }

    # ---------------------------
    # PHASE 4: Check for specialist asking for policy number
    # ---------------------------
    history_lower = conversation_history.lower()
    
    # Look at the last specialist message
    last_specialist_msg = ""
    for marker in ["Billing Agent:", "Policy Agent:", "Claims Agent:", "General Help Agent:"]:
        if marker in conversation_history:
            parts = conversation_history.split(marker)
            if len(parts) > 1:
                last_specialist_msg = parts[-1].split("\n")[0].strip().lower()
                print(f"📝 Last specialist message: {last_specialist_msg[:100]}...")
                break
    
    # If specialist is asking for policy number
    if last_specialist_msg and any(phrase in last_specialist_msg for phrase in [
        "please provide your policy number",
        "provide your policy number",
        "could you provide your policy number",
        "please provide me with your policy number"
    ]):
        print("🚨 Detected: Specialist asking for policy number")
        
        # If we don't have it, ask the user
        if not state.get("policy_number"):
            return {
                "needs_user_input": True,
                "question": "What is your policy number?",
                "missing_info": "policy number",
                "conversation_history": conversation_history,
                "n_iteration": n_iter,
                "policy_number": state.get("policy_number", ""),
                "customer_id": state.get("customer_id", "")
            }

    # ---------------------------
    # PHASE 5: Check for specialist asking for claim ID
    # ---------------------------
    if last_specialist_msg and any(phrase in last_specialist_msg for phrase in [
        "please provide your claim id",
        "provide your claim id",
        "could you provide your claim id"
    ]):
        print("🚨 Detected: Specialist asking for claim ID")
        
        if not state.get("claim_id"):
            return {
                "needs_user_input": True,
                "question": "What is your claim ID?",
                "missing_info": "claim ID",
                "conversation_history": conversation_history,
                "n_iteration": n_iter,
                "policy_number": state.get("policy_number", ""),
                "customer_id": state.get("customer_id", ""),
                "claim_id": state.get("claim_id", "")
            }

    # ---------------------------
    # PHASE 6: Check if question is already answered
    # ---------------------------
    if last_specialist_msg and len(last_specialist_msg) > 0:
        # Check if response has actual information
        if any(indicator in last_specialist_msg for indicator in [
            "$", "premium", "coverage", "deductible", "claim status",
            "approved", "denied", "pending", "amount", "balance"
        ]) and not any(phrase in last_specialist_msg for phrase in [
            "please provide", "could you provide", "can you provide",
            "unable to retrieve", "couldn't find", "couldn't retrieve"
        ]):
            print("✅ Question appears to be answered → routing to final_answer_agent")
            return {
                "next_agent": "final_answer_agent",
                "task": "Finalize response",
                "justification": "Specialist provided answer",
                "conversation_history": conversation_history,
                "n_iteration": n_iter,
                "end_conversation": True,
                "policy_number": state.get("policy_number", ""),
                "customer_id": state.get("customer_id", "")
            }

    # ---------------------------
    # PHASE 7: If we have policy number, route directly
    # ---------------------------
    if state.get("policy_number"):
        print(f"✅ Have policy number: {state['policy_number']} → routing to billing_agent")
        return {
            "next_agent": "billing_agent",
            "task": "Retrieve premium information",
            "justification": "Policy number available",
            "conversation_history": conversation_history,
            "n_iteration": n_iter,
            "policy_number": state["policy_number"],
            "customer_id": state.get("customer_id", "")
        }

    # ---------------------------
    # PHASE 8: Let LLM decide routing
    # ---------------------------
    print("🤖 Calling LLM for initial routing...")

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
                return {
                    "needs_user_input": True,
                    "question": args["question"],
                    "missing_info": args.get("missing_info", ""),
                    "conversation_history": conversation_history,
                    "n_iteration": n_iter,
                    "policy_number": state.get("policy_number", ""),
                    "customer_id": state.get("customer_id", "")
                }

    # Parse routing decision
    try:
        parsed = json.loads(message.content)
    except:
        parsed = {}

    print(f"➡️ Routing to: {parsed.get('next_agent', 'general_help_agent')}")
    
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
    
    # Update state with results
    updated_state = {"messages": [("assistant", result)]}
    
    # Preserve IDs
    if state.get("policy_number"):
        updated_state["policy_number"] = state["policy_number"]
    if state.get("customer_id"):
        updated_state["customer_id"] = state["customer_id"]
    if state.get("claim_id"):
        updated_state["claim_id"] = state["claim_id"]
    
    # Update conversation history
    current_history = state.get("conversation_history", "")
    updated_state["conversation_history"] = current_history + f"\nClaims Agent: {result}"
    
    # ⚠️ CRITICAL FIX: Signal completion
    updated_state["end_conversation"] = True
    updated_state["next_agent"] = "final_answer_agent"
    
    return updated_state
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
    
    # Update state with results
    updated_state = {"messages": [("assistant", result)]}
    
    # Preserve IDs
    if state.get("policy_number"):
        updated_state["policy_number"] = state["policy_number"]
    if state.get("customer_id"):
        updated_state["customer_id"] = state["customer_id"]
    
    # Update conversation history
    current_history = state.get("conversation_history", "")
    updated_state["conversation_history"] = current_history + f"\nPolicy Agent: {result}"
    
    # ⚠️ CRITICAL FIX: Signal completion
    updated_state["end_conversation"] = True
    updated_state["next_agent"] = "final_answer_agent"
    
    return updated_state

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
    
    # ⚠️ CRITICAL FIX: Signal that conversation should end
    updated_state["end_conversation"] = True
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

    # Update conversation history
    updated_state["conversation_history"] = conversation_history + f"\nGeneral Help Agent: {final_answer}"
    
    # ⚠️ CRITICAL FIX: Signal completion
    updated_state["end_conversation"] = True
    updated_state["next_agent"] = "final_answer_agent"

    return updated_state

@trace_agent
def human_escalation_node(state):
    print("---HUMAN ESCALATION AGENT---")
    logger.warning(f"Escalation triggered - State: { {k: v for k, v in state.items() if k != 'messages'} }")
    
    prompt = HUMAN_ESCALATION_PROMPT.format(
        task=state.get("task"),
        conversation_history=state.get("conversation_history", "")
    )

    print("🤖 Generating escalation response...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
    """Determine the next agent based on state"""
    
    # Priority 1: Check for human escalation
    if state.get("requires_human_escalation"):
        print("🚨 Routing to human_escalation_agent")
        return "human_escalation_agent"
    
    # Priority 2: Check if conversation should end
    if state.get("end_conversation"):
        print("✅ Routing to final_answer_agent")
        return "final_answer_agent"
    
    # Priority 3: Check if we need user input
    if state.get("needs_user_input"):
        print("⏸️ Pausing for user input")
        return "supervisor_agent"
    
    # Priority 4: Check if we need clarification
    if state.get("needs_clarification"):
        print("❓ Processing clarification")
        return "supervisor_agent"
    
    # Priority 5: Follow next_agent directive
    next_agent = state.get("next_agent", "general_help_agent")
    print(f"➡️ Routing to: {next_agent}")
    return next_agent









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
