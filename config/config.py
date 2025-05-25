import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "database": os.getenv("DB_NAME", "financial_tracker")
}

# File Paths for Logging & Reports
FILE_PATHS = {
    "logs": os.getenv("LOG_FILE"),
    "reports": os.getenv("REPORTS_FOLDER")
}

# Default App Settings
DEFAULT_PREFERENCES = {
    "currency": os.getenv("DEFAULT_CURRENCY", "USD"),
    "budget_notification": os.getenv("BUDGET_NOTIFICATIONS", "True") == "True"
}