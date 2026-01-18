import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ask_user(question: str, missing_info: str = ""):
    """
    Streamlit-safe version of ask_user.
    Instead of using input(), we return the question so the UI can ask.
    """
    logger.info(f"🗣️ Asking user for input: {question}")

    return {
        "needs_user_input": True,
        "question": question,
        "missing_info": missing_info
    }


def get_policy_details(policy_number: str) -> Dict[str, Any]:
    """Fetch a customer's policy details by policy number"""
    logger.info(f"🔍 Fetching policy details for: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, c.first_name, c.last_name 
        FROM policies p 
        JOIN customers c ON p.customer_id = c.customer_id 
        WHERE p.policy_number = ?
    """, (policy_number,))
    result = cursor.fetchone()
    conn.close()
    if result:
        logger.info(f"✅ Policy found: {policy_number}")
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    logger.warning(f"❌ Policy not found: {policy_number}")
    return {"error": "Policy not found"}

def get_claim_status(claim_id: str = None, policy_number: str = None) -> Dict[str, Any]:
    """Get claim status and details"""
    logger.info(f"🔍 Fetching claim status - Claim ID: {claim_id}, Policy: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    if claim_id:
        cursor.execute("""
            SELECT c.*, p.policy_type 
            FROM claims c
            JOIN policies p ON c.policy_number = p.policy_number
            WHERE c.claim_id = ?
        """, (claim_id,))
    elif policy_number:
        cursor.execute("""
            SELECT c.*, p.policy_type 
            FROM claims c
            JOIN policies p ON c.policy_number = p.policy_number
            WHERE c.policy_number = ?
            ORDER BY c.claim_date DESC LIMIT 3
        """, (policy_number,))
    result = cursor.fetchall()
    conn.close()
    if result:
        logger.info(f"✅ Found {len(result)} claim(s)")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in result]
    logger.warning("❌ No claims found")
    return {"error": "Claim not found"}

def get_billing_info(policy_number: str = None, customer_id: str = None) -> Dict[str, Any]:
    """Get billing information including current balance and due dates"""
    logger.info(f"🔍 Fetching billing info - Policy: {policy_number}, Customer: {customer_id}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    if policy_number:
        cursor.execute("""
            SELECT b.*, p.premium_amount, p.billing_frequency
            FROM billing b
            JOIN policies p ON b.policy_number = p.policy_number
            WHERE b.policy_number = ? AND b.status = 'pending'
            ORDER BY b.due_date DESC LIMIT 1
        """, (policy_number,))
    elif customer_id:
        cursor.execute("""
            SELECT b.*, p.premium_amount, p.billing_frequency
            FROM billing b
            JOIN policies p ON b.policy_number = p.policy_number
            WHERE p.customer_id = ? AND b.status = 'pending'
            ORDER BY b.due_date DESC LIMIT 1
        """, (customer_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        logger.info("✅ Billing info found")
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    logger.warning("❌ Billing info not found")
    return {"error": "Billing information not found"}

def get_payment_history(policy_number: str) -> List[Dict[str, Any]]:
    """Get payment history for a policy"""
    logger.info(f"🔍 Fetching payment history for policy: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.payment_date, p.amount, p.status, p.payment_method
        FROM payments p
        JOIN billing b ON p.bill_id = b.bill_id
        WHERE b.policy_number = ?
        ORDER BY p.payment_date DESC LIMIT 10
    """, (policy_number,))
    
    results = cursor.fetchall()
    conn.close()
    
    if results:
        logger.info(f"✅ Found {len(results)} payment records")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    logger.warning("❌ No payment history found")
    return []

def get_auto_policy_details(policy_number: str) -> Dict[str, Any]:
    """Get auto-specific policy details including vehicle info and deductibles"""
    logger.info(f"🔍 Fetching auto policy details for: {policy_number}")
    conn = sqlite3.connect('insurance_support.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT apd.*, p.policy_type, p.premium_amount
        FROM auto_policy_details apd
        JOIN policies p ON apd.policy_number = p.policy_number
        WHERE apd.policy_number = ?
    """, (policy_number,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        logger.info("✅ Auto policy details found")
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    logger.warning("❌ Auto policy details not found")
    return {"error": "Auto policy details not found"}



