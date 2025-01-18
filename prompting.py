import requests
import json


def get_ollama_response(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/chat"
    prompt = prompt
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    all_response = ""
    response = requests.post(url, json=payload, stream=True)
    # checking if ollama is running
    if response.status_code == 200:
        print("Streaming response from Ollama:\n")
        for line in response.iter_lines(decode_unicode=True):
            if line:  # checking if line is not empty
                try:
                    # Parse each line as a JSON object
                    json_data = json.loads(line)

                    # Extract and print the assistant's message content
                    if "message" in json_data and "content" in json_data["message"]:
                        print(json_data["message"]["content"], end="")
                        all_response += (json_data["message"]["content"])

                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    # return the response as a string for further processing
    return all_response


if __name__ == "__main__":
    prompt = "Provide me a short story about a soldier"
    get_ollama_response(prompt)
