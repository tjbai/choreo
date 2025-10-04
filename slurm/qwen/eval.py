import fire, json, asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from llama import Tokenizer
from llama.util import judge_all
from tqdm.asyncio import tqdm_asyncio

load_dotenv()
client = AsyncOpenAI()

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
        to_judge = [{
            'problem': d['inputs']['problem'],
            'baseline_solution': d['inputs']['solution'],
            'agent_solutions': [
                tokenizer.decode(agent_tokens)
                for agent_tokens in d['outputs'].get('debate_tokens', [[]])[-1]
            ],
        } for d in data]
        is_madpar = True

    elif workflow_type.startswith('mad'):
        to_judge = [{
            'problem': d['inputs']['problem'],
            'baseline_solution': d['inputs']['solution'],
            'test_solution': tokenizer.decode(d['outputs'].get('final_tokens', [[]])[0]),
        } for d in data]
        is_madpar = False

    elif workflow_type.startswith('tot'):
        to_judge = [{
            'problem': d['inputs']['problem'],
            'baseline_solution': d['inputs']['solution'],
            'test_solution': tokenizer.decode(d['outputs'].get('final_tokens', []) or []),
        } for d in data]
        is_madpar = False

    results = asyncio.run(judge_all(to_judge, is_madpar=is_madpar))

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
