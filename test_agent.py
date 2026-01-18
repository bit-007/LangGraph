from agent_app import app   # import your compiled graph

def run_test_query(query):
    """Test the system with a billing query"""
    state = {
        "n_iteraton": 0,
        "messages": [],
        "user_input": query,
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
        "conversation_history": f"User: {query}",
        "task": "Help user with their query",
        "final_answer": ""
    }

    print(f"\n{'='*50}")
    print(f"QUERY: {query}")
    print(f"{'='*50}\n")

    while True:
        state = app.invoke(state)

        if state.get("needs_user_input"):
            print(f"\n🔹 AGENT ASKS: {state['question']}")
            user_answer = input("Your answer: ")

            state = {
                **state,
                "needs_clarification": True,
                "clarification_question": state["question"],
                "user_clarification": user_answer,
                "user_input": user_answer,
            }
            continue

        print("\n---FINAL RESPONSE---")
        print(state.get("final_answer", "No final answer generated."))
        return state
