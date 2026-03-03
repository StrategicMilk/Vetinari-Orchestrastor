"""Test harness for LM Studio chat API."""
import requests


def test_chat_endpoint():
    """Test the LM Studio chat endpoint with the correct payload format."""
    # Use the correct host
    url = "http://100.78.30.7:1234/api/v1/chat"
    payload = {
        "model": "qwen2.5-0.5b-instruct",
        "system_prompt": "You answer only in rhymes.",
        "input": "Hello"
    }
    
    print(f"Testing LM Studio chat endpoint: {url}")
    print(f"Payload: {payload}")
    
    r = requests.post(url, json=payload, timeout=60)
    print(f"Status code: {r.status_code}")
    print(f"Response text: {r.text[:500]}")
    
    if r.status_code != 200:
        print(f"Error response: {r.text}")
        return False
    
    data = r.json()
    print(f"Response: {data}")
    
    if "output" in data:
        print(f"Output: {data['output']}")
        print("Test passed!")
        return True
    else:
        print("Response missing 'output' field")
        return False


if __name__ == "__main__":
    test_chat_endpoint()
