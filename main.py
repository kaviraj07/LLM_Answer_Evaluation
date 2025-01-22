import json
from rouge_score import rouge_scorer
from prompting import get_ollama_response

dataset_path = 'SQuAD/dev-v2.0.json'
count = 0


def score(ground_truth, llm_answer):
    global count
    count += 1
    print(f"scoring question {count}")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, llm_answer)
    return scores['rougeL']


def save_results(results):
    print('Saving results...')
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=2)


def evaluate_llm():
    results = {}
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)
        print('Evaluating LLM...')
        for data in dataset['data']:
            for paragraph in data['paragraphs']:
                for qa in paragraph['qas']:
                    if not qa['is_impossible']:
                        question = qa['question']
                        ground_truth = qa['answers'][0]['text']

                        llm_answer = get_ollama_response(question)
                        rouge_score = score(ground_truth, llm_answer)

                        results[question] = {
                            'ground_truth': ground_truth,
                            'llm_answer': llm_answer,
                            'rouge_score': {'precision': rouge_score.precision,
                                            'recall': rouge_score.recall,
                                            'fmeasure': rouge_score.fmeasure}
                        }

    save_results(results)
    print('Evaluation completed!')


if __name__ == '__main__':
    evaluate_llm()
