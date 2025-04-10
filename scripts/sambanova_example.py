import os
from llms.sambanova import SambaNovaClient

def main():
    # Initialize the client
    client = SambaNovaClient()
    
    # Example usage
    response = client("Hello, how are you?")
    print("Response:", response)
    
    # Example with custom parameters
    response = client.generate(
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.2,
        top_p=0.2
    )
    print("Response with custom parameters:", response)

if __name__ == "__main__":
    main() 