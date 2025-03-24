from typing import Dict
from itertools import permutations
import numpy as np

from datasets import load_dataset, Dataset as HfDataset

from llama import Llama

def load_race(split='train'):
    """
    Load and flatten the RACE dataset for positional bias evaluation.
    Returns a dataset with one question per row instead of multiple questions per article.
    """
    import ast

    ds = load_dataset('EleutherAI/race')
    ds = ds['test'].shuffle(seed=42)

    total_size = len(ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    if split == 'train':
        selected_ds = ds[:train_size]
    elif split == 'val' or split == 'dev':
        selected_ds = ds[train_size:train_size + val_size]
    elif split == 'test':
        selected_ds = ds[train_size + val_size:]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val'/'dev', or 'test'.")

    flattened_data = {
        'article': [],
        'question': [],
        'options': [],
        'answer': []
    }

    for i in range(len(selected_ds['article'])):
        article = selected_ds['article'][i]
        problems = ast.literal_eval(selected_ds['problems'][i])

        for problem in problems:
            flattened_data['article'].append(article)
            flattened_data['question'].append(problem['question'])
            flattened_data['options'].append(problem['options'])
            flattened_data['answer'].append(problem['answer'])

    return HfDataset.from_dict(flattened_data)

def answer_single(
    llama: Llama,
    article: str,
    question: str,
    options: list,
    option_order: list | None = None
) -> str | None:
    system_prompt = "You are solving a multiple-choice question based on a reading passage. Answer with just a single letter (A, B, C, or D) representing your answer choice."

    if option_order is None:
        option_order = list(range(len(options)))

    reordered_options = [options[i] for i in option_order]

    prompt = f"Article: {article}\n\nQuestion: {question}\n\nOptions:\n"
    for i, option in enumerate(reordered_options):
        prompt += f"{chr(65 + i)}. {option}\n"
    prompt += "\nGive your answer as a single letter (A, B, C, or D):"

    dialog = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    prediction = llama.chat_completion([dialog], temperature=0.0, seed=42)[0]
    pred_text = prediction['generation']['content'].strip()

    for char in pred_text:
        if char in "ABCD":
            return char

    return None

def answer_dataset(llama: Llama, dataset: HfDataset) -> Dict:
    answers = []
    correct = []

    for example in dataset:
        article = example['article']
        question = example['question']
        options = example['options']
        true_answer = example['answer']

        pred_answer = answer_single(llama, article, question, options)
        is_correct = pred_answer == true_answer

        answers.append(pred_answer)
        correct.append(is_correct)

    return {
        'answers': answers,
        'correct': correct,
        'accuracy': sum(correct) / len(correct) if correct else 0
    }

def answer_with_permutations(
    llama: Llama,
    dataset: HfDataset,
    num_samples: int | None = None
) -> Dict:
    all_answers = []
    all_correct = []
    permutation_answers = []

    for idx, example in enumerate(dataset):
        article = example['article']
        question = example['question']
        options = example['options']
        true_answer = example['answer']

        orig_order = list(range(len(options)))

        all_perms = list(permutations(orig_order))
        if num_samples and num_samples < len(all_perms):
            np.random.seed(42)
            perm_subset = np.random.choice(len(all_perms), size=num_samples, replace=False)
            selected_perms = [all_perms[i] for i in perm_subset]
        else:
            selected_perms = all_perms

        preds = []
        for perm in selected_perms:
            pred_letter = answer_single(llama, article, question, options, perm)

            if pred_letter:
                letter_idx = ord(pred_letter) - ord('A')
                if 0 <= letter_idx < len(perm):
                    orig_idx = perm[letter_idx]
                    orig_letter = chr(orig_idx + ord('A'))
                    preds.append(orig_letter)
                else:
                    preds.append(None)
            else:
                preds.append(None)

        valid_preds = [p for p in preds if p]
        if valid_preds:
            unique_preds, counts = np.unique(valid_preds, return_counts=True)
            consensus_pred = unique_preds[np.argmax(counts)]
            is_correct = consensus_pred == true_answer
        else:
            consensus_pred = None
            is_correct = False

        all_answers.append(consensus_pred)
        all_correct.append(is_correct)
        permutation_answers.append(preds)

    return {
        'answers': all_answers,
        'correct': all_correct,
        'accuracy': sum(all_correct) / len(all_correct) if all_correct else 0,
        'permutation_answers': permutation_answers
    }
