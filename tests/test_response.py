
import unittest
from google import genai
import sys
sys.path.append(r"D:\@icog_projects\personalized-financial-assistant\config")
from config import GEMINI_API_KEY

class TestGeminiResponse(unittest.TestCase):
    def test_response_not_empty(self):
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_id = "gemini-2.0-flash"
        res = client.models.generate_content(
            model=model_id,
            contents=["Tell me 1 good fact about Nuremberg."]
        )
        response = res.text
        print(response)
        self.assertGreater(len(response), 0, "Response should not be empty")

if __name__ == "__main__":
    unittest.main()