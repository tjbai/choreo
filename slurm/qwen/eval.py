import fire, json, asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from llama import Tokenizer
from tqdm.asyncio import tqdm_asyncio

load_dotenv()
client = AsyncOpenAI()

async def extract_answer(solution: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        prompt = f"""Extract the final numerical answer from this solution. Return only the answer, nothing else.

Solution: {solution}

Respond in JSON:
{{"answer": "the extracted answer"}}"""

        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)["answer"]

async def judge_bo3(
    problem: str,
    baseline_solution: str,
    test_solution: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    baseline_answer, test_answer = await asyncio.gather(
        extract_answer(baseline_solution, semaphore),
        extract_answer(test_solution, semaphore)
    )
    
    async with semaphore:
        prompt = f"""Problem: {problem}

Baseline Answer (assumed correct): {baseline_answer}

Test Answer: {test_answer}

Determine if the test answer is correct beyond reasonable doubt by comparing it to the baseline.
The answers may be in different formats or have minor numerical differences due to rounding, but judge if they're mathematically equivalent.

Respond in JSON:
{{"test_correct": true/false, "reasoning": "brief explanation"}}"""

        responses = await asyncio.gather(*[
            client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            for _ in range(3)
        ])

        judges = [json.loads(r.choices[0].message.content) for r in responses]
        correct_votes = sum(j["test_correct"] for j in judges)

        return {
            "test_correct": correct_votes >= 2,
            "votes": {"correct": correct_votes, "incorrect": 3 - correct_votes},
            "judges": judges,
            "extracted_answers": {"baseline": baseline_answer, "test": test_answer}
        }

async def judge_all(to_judge: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(10)
    tasks = [judge_bo3(**item, semaphore=semaphore) for item in to_judge]
    results = await tqdm_asyncio.gather(*tasks, desc="Judging")
    return results

def main(
    workflow_type: str,
    ft: bool = False,
):
    data = []
    for shard_idx in range(5):
        with open(f'/scratch4/jeisner1/tjbai/qwen_data/{workflow_type}/test/math_shard-{shard_idx}_ft-{ft}.json') as f:
            data.extend(json.load(f))

    tokenizer = Tokenizer('/scratch4/jeisner1/tjbai/qwen3_8b', 'qwen')

    if workflow_type.startswith('madpar'):
        ...

    elif workflow_type.startswith('mad'):
        to_judge = [{
            'problem': d['inputs']['problem'],
            'baseline_solution': d['inputs']['solution'],
            'test_solution': tokenizer.decode(d['outputs'].get('final_tokens', [[]])[0]),
        } for d in data]

    elif workflow_type.startswith('tot'):
        ...

    results = asyncio.run(judge_all(to_judge))

    output = []
    for orig, result in zip(to_judge, results):
        output.append({**orig, 'judgment': result})

    output_path = f'/scratch4/jeisner1/tjbai/qwen_data/{workflow_type}/test/judged_ft-{ft}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    correct = sum(r['test_correct'] for r in results)
    total = len(results)
    print(f"\nResults: {correct}/{total} correct ({100*correct/total:.2f}%)")
    print(f"Output written to: {output_path}")

if __name__ == '__main__':
    fire.Fire(main)
