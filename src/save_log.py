import datetime
import sys
sys.path.append(r"D:\@icog_projects\personalized-financial-assistant\config")  
from config import FILE_PATHS

def log_transaction(user_id, date, amount, payment_method = 'cash', description = 'for unexpected expenses'):
    """
    Logs transaction details inside logs.txt, including user ID, date, amount, payment method, and description. this could be user for backup or debugging purposes if failurity occurs.
    Args:
        user_id (int): Unique ID of the user.
        date (str): Date of the transaction in YYYY-MM-DD format.
        amount (float): Amount of the transaction.
        payment_method (str): Method of payment (default is 'cash').
        description (str): Description of the transaction (default is 'for unexpected expenses').
    Returns:
        str: Success message as that the transaction was logged or error message.
    """
    log_entry = (
        f"{datetime.datetime.now()} | User: {user_id} | Date: {date} "
        f"| Amount: ${amount} | Payment: {payment_method} | Desc: {description}\n"
    )

    # Save log to logs.txt
    with open(FILE_PATHS["logs"], "a") as log_file:
        log_file.write(log_entry)

    return "Transaction successfully logged."