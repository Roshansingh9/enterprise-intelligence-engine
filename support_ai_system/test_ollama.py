"""
Quick Ollama Connection Test
Run: python test_ollama.py
"""

import requests
import time

OLLAMA_ENDPOINT = "http://localhost:11434"

def test_ollama():
    print("=" * 50)
    print("🔍 TESTING OLLAMA CONNECTION")
    print("=" * 50)
    
    # 1. Check if Ollama is running
    print("\n1️⃣ Checking Ollama server...")
    try:
        response = requests.get(f"{OLLAMA_ENDPOINT}/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✅ Ollama server is running")
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to Ollama. Is it running?")
        print("   💡 Run: ollama serve")
        return False
    
    # 2. List available models
    print("\n2️⃣ Available models:")
    data = response.json()
    models = [m['name'] for m in data.get('models', [])]
    if models:
        for m in models:
            print(f"   - {m}")
    else:
        print("   ⚠️ No models found!")
        print("   💡 Run: ollama pull phi3")
        return False
    
    # 3. Check for recommended models
    print("\n3️⃣ Checking for memory-efficient models...")
    small_models = ['phi3', 'tinyllama', 'gemma:2b', 'phi3:mini']
    found = [m for m in models if any(s in m for s in small_models)]
    if found:
        print(f"   ✅ Found: {found}")
        test_model = found[0]
    else:
        print("   ⚠️ No small models found. You may run out of memory!")
        print("   💡 Run: ollama pull phi3")
        if models:
            test_model = models[0]
        else:
            return False
    
    # 4. Test a simple generation
    print(f"\n4️⃣ Testing generation with '{test_model}'...")
    print("   (This may take 30-60 seconds on first run)")
    
    start = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json={
                "model": test_model,
                "prompt": "Say 'Hello, I am working!' in exactly 5 words.",
                "stream": False,
                "options": {
                    "num_predict": 20,
                    "temperature": 0.1
                }
            },
            timeout=120
        )
        
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '').strip()
            print(f"   ✅ Response received in {elapsed:.1f}s")
            print(f"   📝 Output: {text[:100]}")
            return True
        else:
            print(f"   ❌ Error {response.status_code}: {response.text[:200]}")
            if "memory" in response.text.lower():
                print("   💡 Model too large! Try: ollama pull tinyllama")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out after {time.time() - start:.0f}s")
        print("   💡 Model might be loading or system is overloaded")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama()
    print("\n" + "=" * 50)
    if success:
        print("✅ OLLAMA IS WORKING!")
        print("You can now run: python main.py train")
    else:
        print("❌ OLLAMA NEEDS ATTENTION")
        print("Fix the issues above and try again.")
    print("=" * 50)
