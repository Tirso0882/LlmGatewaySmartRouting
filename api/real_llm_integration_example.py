"""
Example: Where Real LLM Calls Would Be Integrated
This shows how to replace mock responses with real OpenAI API calls
"""

import os
import time
from typing import Tuple

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: dotenv not installed. Please install it using 'pip install python-dotenv'")
    exit(1)

load_dotenv()


class RealLLMIntegration:
    """Example of how real LLM calls would be integrated"""
    
    def __init__(self):
        # Azure OpenAI configuration
        self.api_key = os.getenv('O3_API_KEY')
        self.endpoints = {
            'o3': os.getenv('O3_ENDPOINT'),
            'gpt-4o-mini': os.getenv('GPT4O_MINI_ENDPOINT'),
            'o4-mini': os.getenv('O4_MINI_ENDPOINT')
        }
    
    def call_real_llm(self, model_name: str, prompt: str) -> Tuple[str, float]:
        """Make real API call to Azure OpenAI"""
        
        start_time = time.time()
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key
        }
        
        if model_name in ['o3', 'o4-mini']:
            payload = {
                'messages': [{'role': 'user', 'content': prompt}],
            }
        else:  # gpt-4o-mini
            payload = {
                'messages': [{'role': 'user', 'content': prompt}],
            }
        
        response = requests.post(
            self.endpoints[model_name],
            headers=headers,
            json=payload
        )
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['choices'][0]['message']['content']
            return llm_response, response_time
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")


def test_real_llm_integration():
    """Test function to demonstrate the RealLLMIntegration class"""
    print("Testing Real LLM Integration...")
    
    required_env_vars = ['O3_API_KEY', 'O3_ENDPOINT', 'GPT4O_MINI_ENDPOINT', 'O4_MINI_ENDPOINT']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set up your .env file with the required Azure OpenAI credentials.")
        return
    
    try:
        real_llm = RealLLMIntegration()
        
        test_prompt = "Hello! Can you tell me a short joke?"
        
        print(f"✅ Environment variables loaded successfully")
        print(f"📝 Test prompt: {test_prompt}")
        print(f"🔧 Available models: {list(real_llm.endpoints.keys())}")
        
        available_models = list(real_llm.endpoints.keys())
        if available_models:
            test_model = available_models[0]
            print(f"\n🚀 Testing with model: {test_model}")
            
            try:
                response, response_time = real_llm.call_real_llm(test_model, test_prompt)
                print(f"✅ Success! Response time: {response_time:.2f}ms")
                print(f"🤖 LLM Response: {response}")
            except Exception as e:
                print(f"❌ Error calling LLM: {str(e)}")
                print("This might be due to:")
                print("- Invalid API key")
                print("- Incorrect endpoint URL")
                print("- Network connectivity issues")
                print("- Azure OpenAI service not available")
        else:
            print("❌ No models configured")
            
    except Exception as e:
        print(f"❌ Error initializing RealLLMIntegration: {str(e)}")


if __name__ == "__main__":
    test_real_llm_integration()

"""
# Replace this line in main.py:
llm_response, llm_response_time = mock_llm.generate_response(recommended_model, request.prompt)

# With this:
real_llm = RealLLMIntegration()
llm_response, llm_response_time = real_llm.call_real_llm(recommended_model, request.prompt)
"""
