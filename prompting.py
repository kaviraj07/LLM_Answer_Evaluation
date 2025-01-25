import requests
import json
import ollama

from ollama import chat, ChatResponse

# from ollama_serve import start_ollama, stop_ollama
import asyncio


# Chattting with LLM API
def stream_ollama_response(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/chat"
    prompt = prompt
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
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
                        all_response += json_data["message"]["content"]

                except json.JSONDecodeError:
                    print(f"\nFailed to parse line: {line}")
        print()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    # return the response as a string for further processing
    return all_response


# Generate with LLM API Synchronosly
def sync_get_ollama_response(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    prompt = prompt + "only the answer is needed."
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload)
    # checking if ollama is running
    if response.status_code == 200:
        answer = response.json()["response"]
        return answer
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


# Generate with LMM API Asynchronosly
async def get_ollama_response(session, prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    prompt = prompt + "only the answer is needed."
    payload = {"model": model, "prompt": prompt, "stream": False}
    async with session.post(url, json=payload) as response:
        if response.status == 200:
            data = await response.json()
            return data["response"]
        else:
            print(f"Error: {response.status}")
            print(await response.text())
            return None


# Chat with Ollama SDK
def chatting(question):
    question = question + "only the answer is needed."
    try:
        response: ChatResponse = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        print(response["message"]["content"])
        print(response.message.content)
        return response.message.content
    except ConnectionError as e:
        print(f"Failed to connect to Ollama: {e}")


# Generate with Ollama SDK
async def generate(question, client, model="llama3.2"):
    question = question + "only the answer is needed."

    response = await client.generate(model, question)
    # print(response["response"])
    return response["response"]


async def main():
    print("Inside main")
    # Dictionary of questions
    questions = {
        "question1": "What is the capital of France?",
        "question2": "What is the capital of India?",
        "question3": "What is the capital of China?",
        "question4": "What is the capital of Japan?",
        "question5": "What is the capital of Russia?",
    }
    client = ollama.AsyncClient()
    # Create a list of tasks to run concurrently
    tasks = [generate(q, client) for q in questions.values()]

    # Run all tasks concurrently and gather results in order
    results = await asyncio.gather(*tasks)

    # Print results in the order of the questions
    for i, (key, question) in enumerate(questions.items()):
        print(f"{key}: {question}")
        print(f"Answer: {results[i]}\n")


if __name__ == "__main__":
    asyncio.run(main())
