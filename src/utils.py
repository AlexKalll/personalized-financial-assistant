import datetime
import sys
sys.path.append(r"D:\@icog_projects\personalized-financial-assistant\config")  
from config import FILE_PATHS

def log_transaction(user_id, date, amount, category, payment_method, description):
    """Logs transaction details inside logs.txt."""
    log_entry = (
        f"{datetime.datetime.now()} | User: {user_id} | Date: {date} "
        f"| Amount: ${amount} | Category: {category} | Payment: {payment_method} | Desc: {description}\n"
    )

    # Save log to logs.txt
    with open(FILE_PATHS["logs"], "a") as log_file:
        log_file.write(log_entry)

    return "Transaction successfully logged."