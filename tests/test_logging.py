import unittest
from datetime import datetime
import sys
sys.path.append(r"D:\@icog_projects\personalized-financial-assistant\src")
from save_log import log_transaction

class TestLogTransaction(unittest.TestCase):
    def test_log_transaction(self):
        user_id = 1
        date = datetime.now().strftime("%Y-%m-%d")
        amount = 100.0
        payment_method = "cash"
        description = "for unexpected expenses"
        result = log_transaction(user_id, date, amount, payment_method, description)
        self.assertEqual(result, "Transaction successfully logged.")

if __name__ == "__main__":
    unittest.main()