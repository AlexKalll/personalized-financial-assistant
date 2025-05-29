import streamlit as st
from typing import Dict, Union, Optional
import os
import sys
import re
import pandas as pd
import datetime
from fpdf import FPDF
from google import genai
from google.genai.types import GenerateContentConfig

# Add config and data paths to sys.path
sys.path.append(r"D:\@icog_projects\personalized-financial-assistant\config")
sys.path.append(r"D:\@icog_projects\personalized-financial-assistant\data")

from db_connection import create_connection
from config import GEMINI_API_KEY, DB_CONFIG, FILE_PATHS
from save_log import log_transaction

# gemini Setup
client = genai.Client(api_key=GEMINI_API_KEY)
model_id = "gemini-1.5-flash"

# --- Tools for Gemini ---
def retrieve_user_data(user_id: int) -> Union[Dict[str, Union[int, str, float, None]], Dict[str, str]]:
    """
    Retrieves a user's profile from the database, including personal details and salary.

    Args:
        user_id (int): The unique identifier of the user.

    Returns:
        dict: A dictionary containing user details (user_id, fname, lname, age, gender,
              occupation, email, created_at, salary) or an error message if the user is not found.
    """
    db = create_connection()
    if not db:
        return {"error": "Database connection failed."}
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT u.user_id, u.fname, u.lname, u.age, u.gender, u.occupation, u.email, u.created_at, b.salary
        FROM Users u
        LEFT JOIN Budgets b ON u.user_id = b.user_id
        WHERE u.user_id = %s AND b.month = 'May' 
    """, (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    db.close()

    if not user_data:
        return {"error": "User not found."}

    return {
        "user_id": user_data["user_id"],
        "fname": user_data["fname"],
        "lname": user_data["lname"],
        "age": user_data["age"],
        "gender": user_data["gender"],
        "occupation": user_data["occupation"],
        "email": user_data["email"],
        "created_at": user_data["created_at"].isoformat(),
        "salary": float(user_data["salary"]) 
    }

def analyze_spending(user_id: int) -> Union[Dict, Dict[str, str]]:
    """
    Analyzes spending patterns over the last 3 months and generates a visualization.

    This function retrieves transactions for the specified user, calculates total spending,
    monthly breakdowns, and top payment methods, and creates a bar chart visualizing monthly
    spending. The chart is saved to the reports directory.

    Args:
        user_id (int): The unique identifier of the user.

    Returns:
        dict: A dictionary containing spending insights (total_spent, monthly_breakdown,
              top_payment_methods) or an error message if no transactions are found.

    Example:
        >>> analyze_spending(1)
        {
            'total_spent': 1500.0,
            'monthly_breakdown': {'March': 500.0, 'April': 600.0, 'May': 400.0},
            'top_payment_methods': {'CBE': 10, 'Cash': 5}
        }
    """
    db = create_connection()
    if not db:
        return {"error": "Database connection failed."}
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT date, amount, payment_method, description 
        FROM Transactions 
        WHERE user_id = %s AND date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
        ORDER BY date ASC
    """, (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    db.close()

    if not transactions:
        return {"error": "No transactions found in the last 3 months."}

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%B")

    total_spent = float(df["amount"].sum())  # Convert Decimal to float
    monthly_spent = {month: float(amount) for month, amount in df.groupby("month")["amount"].sum().items()}  # Convert Decimal to float
    top_payment_methods = df["payment_method"].value_counts().to_dict()

    insights = {
        "total_spent": total_spent,
        "monthly_breakdown": monthly_spent,
        "top_payment_methods": top_payment_methods
    }

    return insights

def generate_financial_advice(user_id: int) -> Dict[str, Union[float, str]]:
    """
    Generates personalized financial advice based on current financial data.

    This function retrieves the user's budget (salary, expense limit, savings goal) for May
    and recent spending data (last 1 month) to provide tailored financial insights.

    Args:
        user_id (int): The unique identifier of the user.

    Returns:
        dict: A dictionary containing financial data (name, salary, expense_limit,
              savings_goal, total_spent) or an error message if data is missing.

    Example:
        >>> generate_financial_advice(1)
        {
            'name': 'John Doe',
            'salary': 50000.0,
            'expense_limit': 40000.0,
            'savings_goal': 10000.0,
            'total_spent': 30000.0
        }
    """
    db = create_connection()
    if not db:
        return {"error": "Database connection failed."}

    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT salary, expense_limit, savings_goal 
        FROM Budgets 
        WHERE user_id = %s AND month = 'May'
    """, (user_id,))
    budget = cursor.fetchone()
    cursor.execute("""
        SELECT SUM(amount) AS total_spent 
        FROM Transactions 
        WHERE user_id = %s AND date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
    """, (user_id,))
    spending_data = cursor.fetchone()
    cursor.execute("SELECT fname, lname FROM Users WHERE user_id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    db.close()

    if not budget or not user_data:
        return {"error": "No budget or user data found."}

    return {
        "name": f"{user_data['fname']} {user_data['lname']}",
        "salary": float(budget["salary"]),  # Convert Decimal to float
        "expense_limit": float(budget["expense_limit"]),
        "savings_goal": float(budget["savings_goal"]),
        "total_spent": float(spending_data["total_spent"] or 0)
    }

def predict_future_spending(user_id: int) -> Dict[str, float]:
    """
    Predicts future spending based on historical transaction data.

    This function analyzes the last 3 months of transactions to calculate average monthly
    spending and predicts next month's spending with a random variation (90–120%).

    Args:
        user_id (int): The unique identifier of the user.

    Returns:
        dict: A dictionary containing average_spending and predicted_spending, or an error
              message if insufficient data is available.

    Example:
        >>> predict_future_spending(1)
        {
            'average_spending': 500.0,
            'predicted_spending': 550.0
        }
    """
    import numpy as np
    db = create_connection()
    if not db:
        return {"error": "Database connection failed."}

    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT date, amount 
        FROM Transactions 
        WHERE user_id = %s AND date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
        ORDER BY date ASC
    """, (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    db.close()

    if not transactions:
        return {"error": "Not enough transaction history."}

    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%B")

    monthly_totals = df.groupby("month")["amount"].sum()
    avg_spending = float(np.mean(monthly_totals))  # Convert Decimal to float
    predicted_spending = avg_spending * np.random.uniform(0.9, 1.2)

    return {
        "average_spending": avg_spending,
        "predicted_spending": float(predicted_spending)
    }


def record_transaction(user_input: str) -> Dict[str, Union[int, float, str]]:
    """
    Records a transaction in the database based on user input.

    This function parses a natural language input to extract transaction details, inserts
    the transaction into the Transactions table, and logs it to a file.

    Args:
        user_input (str): A string describing the transaction, e.g., "User 7 spent 250 ETB for groceries via CBE today".

    Returns:
        dict: A dictionary containing transaction details (user_id, date, amount,
              payment_method, description) or an error message if the input is invalid.

    Example:
        >>> record_transaction("User 7 spent 250 ETB for groceries via CBE today")
        {
            'user_id': 7,
            'date': '2025-05-27',
            'amount': 250.0,
            'payment_method': 'CBE',
            'description': 'groceries'
        }
    """
    db = create_connection()
    if not db:
        return {"error": "Database connection failed."}

    cursor = db.cursor()
    match = re.search(r"User (\d+) spent (\d+\.?\d*) ETB for (.+) via (\w+) today", user_input)
    if not match:
        cursor.close()
        db.close()
        return {"error": "Invalid format. Use: 'User [ID] spent [Amount] ETB for [Purpose] via [Payment Method] today.'"}

    user_id = int(match.group(1))
    amount = float(match.group(2))
    description = match.group(3)
    payment_method = match.group(4)
    date = datetime.date.today().strftime("%Y-%m-%d")

    query = """
        INSERT INTO Transactions (user_id, date, amount, payment_method, description) 
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(query, (user_id, date, amount, payment_method, description))
    db.commit()
    cursor.close()
    db.close()
    # Log the transaction to a file
    log_transaction(user_id, date, amount, payment_method, description)

    return {
        "user_id": user_id,
        "date": date,
        "amount": amount,
        "payment_method": payment_method,
        "description": description
    }

def generate_transaction_receipt(transaction_id: int) -> Dict[str, Union[int, float, str]]:
    """
    Generates a PDF receipt for a specific transaction.

    This function retrieves transaction details from the database, creates a formatted PDF
    receipt, and saves it to the receipts directory.

    Args:
        transaction_id (int): The unique identifier of the transaction.

    Returns:
        dict: A dictionary containing receipt details (user_full_name, transaction_id, user_id, date,
              amount, payment_method, description, receipt_path) or an error message.

    Example:
        >>> generate_transaction_receipt(123)
        {   'user_full_name': 'John Doe',
            'transaction_id': 123,
            'user_id': 7,
            'date': '2025-05-27',
            'amount': 250.0,
            'payment_method': 'CBE',
            'description': 'groceries',
            'receipt_path': 'workflows/receipts/user7-transaction123-receipt.pdf'
        }
    """
    db = create_connection()
    if not db:
        return {"error": "Database connection failed."}

    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Transactions WHERE transaction_id = %s", (transaction_id,))
    transaction = cursor.fetchone()
    cursor.close()

    if not transaction:
        return {"error": f"No transaction found with ID {transaction_id}."}

    user_id = transaction["user_id"]
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT fname, lname FROM Users WHERE user_id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    db.close()
    user_full_name = f"{user['fname']} {user['lname']}"
    date = transaction["date"].strftime("%Y-%m-%d")  # Convert date to string
    amount = float(transaction["amount"])  # Convert Decimal to float
    payment_method = transaction["payment_method"]
    description = transaction["description"]

    receipt_filename = f"workflows/receipts/user{user_id}-transaction{transaction_id}-receipt.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(200, 10, "Transaction Receipt", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"User: {user_full_name}", ln=True)
    pdf.cell(200, 10, f"Transaction ID: {transaction_id}", ln=True)
    pdf.cell(200, 10, f"User ID: {user_id}", ln=True)
    pdf.cell(200, 10, f"Date: {date}", ln=True)
    pdf.cell(200, 10, f"Amount: ETB {amount:.2f}", ln=True)
    pdf.cell(200, 10, f"Payment Method: {payment_method}", ln=True)
    pdf.cell(200, 10, f"Description: {description}", ln=True)
    pdf.cell(200, 10, "", ln=True)
    pdf.cell(200, 10, "", ln=True)
    pdf.cell(200, 10, "Sincerly,", ln=True)
    pdf.cell(200, 10, "Dr. Abebe", ln=True)
    pdf.cell(200, 10, f"{payment_method} Manager", ln=True)
    pdf.output(receipt_filename)

    return {
        "user_full_name": user_full_name,
        "transaction_id": transaction_id,
        "user_id": user_id,
        "date": date,
        "amount": amount,
        "payment_method": payment_method,
        "description": description,
        "receipt_path": receipt_filename
    }

# --- Function Declarations for Gemini ---
functions = [
    retrieve_user_data,
    analyze_spending,
    generate_financial_advice,
    predict_future_spending,
    record_transaction,
    generate_transaction_receipt
]

# --- Configuration and tools for Gemini ---
config = GenerateContentConfig(
    system_instruction="""
    You are an expert financial assistant powered by real-time data from a MySQL database. Your task is to deliver precise, engaging, and personalized financial insights that empower users to make smart decisions. Use the provided tools to fetch and analyze data, analyze spending patterns, predict future spending, generate transaction receipts and provide tailored financial advice. Your responses should be informative and present responses in a visually appealing markdown format with emojis, tables, and bullet points where appropriate. Ensure answers conversational, and tailored to the user's query. 
    """,
    tools = functions
)

# --- Helper Functions ---
def extract_user_id(text: str) -> Optional[int]:
    id = re.search(r"\d+", text)
    
    if id:
        return int(id.group(0))
    return None
    
def extract_transaction_id(text: str) -> Optional[int]:
    tr_id = re.search(r"\d+", text)
    if tr_id:
        return int(tr_id.group(0))
    return None

# --- Streamlit Pages ---
def home():
    st.title("Welcome to Real-Time Financial Assistant")
    st.markdown("""
    ### This assistant helps you:
    - View User\'s profile
    - Analyze spending patterns
    - Get tailored financial advice
    - Predict future spending critically
    - Record transactions to the database
    - Generate receipts for transactions

    Navigate from the sidebar to get started!
    """)

def user_profile_page():
    
    st.title("👤 User Profile Explorer")
    st.caption("Try queries like: `show me the profile of user 1` or `details of user id 3`")

    prompt = st.text_input("Enter a query to show a user profile:")
    if st.button("Show Profile"):
        user_id = extract_user_id(prompt)
        if not user_id:
            st.error("Could not extract user ID. Try: `details user id 2`.")
            return

        with st.spinner("Fetching profile..."):
            gemini_response = client.models.generate_content(
                model=model_id,
                config=config,
                contents=f"""
                Based on the prompt: '{prompt}'. Display the profile of user {user_id} in a vibrant, organized format. Use markdown with:
                - A header with the user's full name 
                - A table for details (User Name, User ID, Age, Gender, Occupation, Email, Account Created, Salary)
                - Emojis for visual appeal
                - A brief summary of the user's financial health
                """
            )
            st.markdown("### Gemini’s Response")
            st.success(gemini_response.text)

def spending_analysis_page():
    """
    Renders the spending analysis page.

    Allows users to analyze spending patterns and displays a chart if available.
    """
    st.title(" Spending Analyser ")
    st.caption("Try queries like: `Analyze user 3` or `Spending pattern for user 5`")

    prompt = st.text_input(" Enter your spending query:")
    if st.button("Analyze Spending"):
        user_id = extract_user_id(prompt)
        if not user_id:
            st.error("Could not extract user ID.")
            return

        with st.spinner("Analyzing spending..."):
            gemini_response = client.models.generate_content(
                model=model_id,
                config=config,
                contents=f"""
                Based on the prompt: '{prompt}'. Analyze the spending patterns for user {user_id} over the last 3 months. Provide a detailed response in markdown format with:
                - A header summarizing total spending
                - A table showing monthly breakdown (Month, Amount in ETB)
                - A list of top payment methods with counts
                - Key insights or trends
                - Emojis for engagement
                Chart is available at: workflows/reports/user{user_id}-spending_chart.png
                """
            )
            st.markdown("### Gemini’s Response")
            st.success(gemini_response.text)

            # Display chart if generated
            chart_path = f"workflows/reports/user{user_id}-spending_chart.png"
            if os.path.exists(chart_path):
                st.image(chart_path, caption="Monthly Spending Breakdown")

def financial_advice_page():
    """
    Renders the financial advice page.

    Provides personalized financial advice based on user data.
    """
    st.title(" Smart Financial Advice")
    st.caption("Try queries like: `Advice for user 2` or `Financial tips for user 4`")

    prompt = st.text_input(" Enter your advice query:")
    if st.button("Get Advice"):
        user_id = extract_user_id(prompt)
        if not user_id:
            st.error("Could not extract user ID.")
            return

        with st.spinner("Generating advice..."):
            gemini_response = client.models.generate_content(
                model=model_id,
                config=config,
                contents=f"""
                Based on the prompt: '{prompt}'. Provide personalized financial advice for user {user_id} using their current financial data. Deliver a scientific yet accessible response in markdown format with:
                - A header with the user's name
                - A table summarizing financial data (Salary, Expense Limit, Savings Goal, Total Spent)
                - Bullet points with actionable advice
                - Emojis for visual flair
                - A motivational closing statement
                """
            )
            st.markdown("### Gemini’s Response")
            st.success(gemini_response.text)

def future_spending_page():
    """
    Renders the future spending prediction page.

    Predicts next month's spending based on historical data.
    """
    st.title("Future Spending Forecast")
    st.caption("Try queries like: `Predict spending for user 1`")

    prompt = st.text_input(" Enter your prediction query:")
    if st.button("Predict Spending"):
        user_id = extract_user_id(prompt)
        if not user_id:
            st.error("Could not extract user ID.")
            return

        with st.spinner("Predicting spending..."):
            gemini_response = client.models.generate_content(
                model=model_id,
                config=config,
                contents=f"""
                Based on the prompt: '{prompt}'. Predict future spending for user {user_id} for the next month using the last 3 months' data. Provide a detailed response in markdown format with:
                - A header stating the prediction
                - A table comparing average and predicted spending
                - Bullet points explaining the prediction methodology
                - Emojis for engagement
                - A planning tip for the user
                """
            )
            st.markdown("### Gemini’s Response")
            st.success(gemini_response.text)

def record_transaction_page():
    """
    Renders the transaction recording page.

    Allows users to log transactions with natural language input.
    """
    st.title("Transaction Recorder")
    st.caption("Enter something like: `User 7 spent 250 ETB for groceries via CBE today`")

    prompt = st.text_input(" Enter transaction details:")
    if st.button("Record Transaction"):
        with st.spinner("Recording transaction..."):
            gemini_response = client.models.generate_content(
                model=model_id,
                config=config,
                contents=f"""
                Based on the prompt: '{prompt}'. Record the transaction in the database and log file. Provide a response in markdown format with:
                - A confirmation header
                - A table summarizing the transaction (User ID, Date, Amount, Payment Method, Description)
                - A SQL query to verify the transaction
                - Emojis for excitement
                """
            )
            st.markdown("### Gemini’s Response")
            st.success(gemini_response.text)

def transaction_receipt_page():
    """
    Renders the transaction receipt generation page.

    Allows users to generate a PDF receipt for a specific transaction ID.
    """
    st.title(" Transaction Receipt Generator")
    st.caption("Enter something like: `Generate a transaction receipt transaction id 123`")

    prompt = st.text_input(" Enter receipt query:")
    if st.button("Generate Receipt"):
        transaction_id = extract_transaction_id(prompt)
        if not transaction_id:
            st.error("Please provide a transaction ID (e.g., `Generate a transaction receipt transaction id 123`).")
            return

        with st.spinner("Generating receipt..."):
            gemini_response = client.models.generate_content(
                model=model_id,
                config=config,
                contents=f"""
                Based on the prompt: '{prompt}'. Generate a receipt for transaction ID {transaction_id}. Provide a response in markdown format with:
                - A confirmation header
                - A table summarizing receipt details (User full name, Transaction ID, User ID, Date, Amount, Payment Method, Description)
                - The file path where the PDF is saved
                - Emojis for professionalism
                File saved at: workflows/receipts/userX-transaction{transaction_id}-receipt.pdf
                """
            )
            st.markdown("###  Gemini’s Response")
            st.success(gemini_response.text)

            # Provide download link for PDF
            receipt_path = f"workflows/receipts/userX-transaction{transaction_id}-receipt.pdf"
            if os.path.exists(receipt_path):
                with open(receipt_path, "rb") as file:
                    st.download_button(
                        label="Download Receipt PDF",
                        data=file,
                        file_name=os.path.basename(receipt_path),
                        mime="application/pdf"
                    )

# --- Pages of the App ---
PAGES = {
    "🏠 Home": home,
    "👤 User Profile": user_profile_page,
    "📊 Spending Analysis": spending_analysis_page,
    "💡 Financial Advice": financial_advice_page,
    "🔮 Future Spending": future_spending_page,
    "💸 Record Transaction": record_transaction_page,
    "📜 Generate Receipts": transaction_receipt_page,
}

def main():
    """
    Main function to run the Streamlit application.

    Configures the page layout and handles navigation between pages.
    """
    st.set_page_config(page_title="Smart Financial Assistant", layout="wide")
    st.sidebar.title("📘 Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()