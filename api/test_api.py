"""
Test script for the LLM Gateway API
Demonstrates the fine-tuned routing capabilities
"""

import json
import time

import requests

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_models():
    """Test the models endpoint"""
    print("🔍 Testing Models Endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Available Models: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_routing():
    """Test single prompt routing"""
    print("🔍 Testing Single Prompt Routing...")
    
    test_prompts = [
        # Simple query - should route to gpt-4o-mini
        "What is the weather like today?",
        
        # Complex reasoning - should route to o3
        "Can you solve this complex mathematical equation: ∫(x² + 2x + 1)dx and explain the steps?",
        
        # Moderate complexity - should route to o4-mini
        "Write a Python function to sort a list of dictionaries by a specific key",
        
        # Code-related - should route to o3
        "Debug this JavaScript code and explain what's wrong: function add(a,b) { return a + b; } console.log(add('5', 3));",
        
        # Creative writing - should route to o4-mini
        "Write a short story about a robot learning to paint"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {prompt[:50]}... ---")
        
        payload = {"prompt": prompt}
        response = requests.post(f"{BASE_URL}/route", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recommended Model: {result['recommended_model']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
            print(f"   Reasoning: {result['reasoning']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    print()

def test_batch_routing():
    """Test batch prompt routing"""
    print("🔍 Testing Batch Routing...")
    
    batch_prompts = [
        "What is 2+2?",
        "Explain quantum computing in detail",
        "Write a haiku about programming",
        "Debug this SQL query: SELECT * FROM users WHERE name = 'John'",
        "Translate 'Hello world' to Spanish"
    ]
    
    payload = {"prompts": batch_prompts}
    response = requests.post(f"{BASE_URL}/route/batch", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Batch Processing Complete!")
        print(f"   Total Time: {result['total_time_ms']:.2f}ms")
        print(f"   Average Time per Prompt: {result['total_time_ms']/len(batch_prompts):.2f}ms")
        
        for i, routing_result in enumerate(result['results'], 1):
            print(f"   {i}. '{batch_prompts[i-1][:30]}...' → {routing_result['recommended_model']} ({routing_result['confidence']:.2%})")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
    
    print()

def test_stats():
    """Test the stats endpoint"""
    print("🔍 Testing Stats Endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Stats: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests"""
    print("🚀 LLM Gateway API Test Suite")
    print("=" * 50)
    
    # Wait a moment for the API to be ready
    print("Waiting for API to be ready...")
    time.sleep(2)
    
    try:
        test_health()
        test_models()
        test_single_routing()
        test_batch_routing()
        test_stats()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on http://localhost:8000")
        print("   Start the API with: python api/main.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()


