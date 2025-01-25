import json
from rouge_score import rouge_scorer
import asyncio
import ollama

import aiofiles

dataset_path = "SQuAD/dev-v2.0.json"


# Generate with Ollama SDK
async def generate(context, question, client, model="llama3.2"):
    question = f"Given the context, provide only the simple answer (no need for full sentence or phrase) based from the context, without any end punction :\nContext:\n{context}\nQuestion:\n{question}"

    response = await client.generate(model, question)
    return response["response"]


def score(ground_truth, llm_answer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(ground_truth, llm_answer)
    return scores["rougeL"]


async def async_save_results(results):
    print("Saving results...")
    results_json = json.dumps(results, indent=2)
    async with aiofiles.open("results.json", "w") as file:
        await file.write(results_json)


async def evaluate_llm():
    results = {}
    client = ollama.AsyncClient()

    # Asynchronously read the dataset file
    async with aiofiles.open(dataset_path, mode="r") as file:
        content = await file.read()
        dataset = json.loads(content)
        print("Evaluating LLM asynchronously...")
        global count
        tasks = []
        questions = []
        ground_truths = []

        for data in dataset["data"]:
            for paragraph in data["paragraphs"]:
                for qa in paragraph["qas"]:
                    if not qa["is_impossible"]:
                        context = paragraph["context"]
                        question = qa["question"]
                        ground_truth = qa["answers"][0]["text"]

                        questions.append(question)
                        ground_truths.append(ground_truth)
                        tasks.append(generate(context, question, client))

    # Run all LLM queries concurrently
    llm_answers = await asyncio.gather(*tasks)

    print("scoring questions")
    for question, ground_truth, llm_answer in zip(
        questions, ground_truths, llm_answers
    ):
        if llm_answer:
            rouge_score = score(ground_truth, llm_answer)
            results[question] = {
                "ground_truth": ground_truth,
                "llm_answer": llm_answer,
                "rouge_score": {
                    "precision": rouge_score.precision,
                    "recall": rouge_score.recall,
                    "fmeasure": rouge_score.fmeasure,
                },
            }
    # Save results asynchronously
    await async_save_results(results)
    print("Evaluation completed!")


async def main():
    await evaluate_llm()


if __name__ == "__main__":
    asyncio.run(main())
