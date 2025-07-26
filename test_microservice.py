import requests
import time

# Base URL for the microservice
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check response: {response.status_code} - {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the microservice. Make sure it's running.")
    except Exception as e:
        print(f"Error: {e}")

def test_embed(text):
    """Test the embedding endpoint"""
    print(f"\nTesting embedding for text: '{text}'")
    try:
        response = requests.post(f"{BASE_URL}/embed", json={"text": text})
        print(f"Embedding response: {response.status_code}")
        if response.status_code == 201:
            print(f"Response: {response.json()}")
            return response.json().get("embedding_id")
        else:
            print(f"Error: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the microservice. Make sure it's running.")
    except Exception as e:
        print(f"Error: {e}")
    return None

def test_search(text, k=3):
    """Test the search endpoint"""
    print(f"\nTesting search for text: '{text}'")
    try:
        response = requests.post(f"{BASE_URL}/search", json={"text": text, "k": k})
        print(f"Search response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result['results'])} results:")
            for i, res in enumerate(result['results']):
                print(f"  {i+1}. Text: {res['text']}")
                print(f"     Distance: {res['distance']}")
                print(f"     ID: {res['id']}")
        else:
            print(f"Error: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the microservice. Make sure it's running.")
    except Exception as e:
        print(f"Error: {e}")

def test_get_all_texts():
    """Test the get all texts endpoint"""
    print("\nTesting get all texts...")
    try:
        response = requests.get(f"{BASE_URL}/texts")
        print(f"Get all texts response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Total texts stored: {result['count']}")
            for i, text in enumerate(result['texts']):
                print(f"  {i}. {text}")
        else:
            print(f"Error: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the microservice. Make sure it's running.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("Embedding Microservice Test Script")
    print("=" * 40)

    # Test health check
    test_health()

    # If health check fails, exit
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("Microservice is not healthy. Exiting.")
            return
    except:
        print("Could not connect to microservice. Please start it first:")
        print("  python app.py")
        return

    # Test embedding
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Vector databases are optimized for similarity search",
        "Natural language processing helps computers understand text"
    ]

    embedding_ids = []
    for text in texts:
        embedding_id = test_embed(text)
        if embedding_id is not None:
            embedding_ids.append(embedding_id)
        time.sleep(0.5)  # Small delay between requests

    # Test search
    test_search("artificial intelligence and machine learning")
    test_search("programming languages")
    test_search("database technology")

    # Test get all texts
    test_get_all_texts()

    print("\nTest completed!")

if __name__ == "__main__":
    main()