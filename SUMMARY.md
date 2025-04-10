# Summary of results

## Information Leakage

### Prisoner’s Dilemma

We consider a conversational game between 2 agents, Alice and Bob, which has 3 steps:
1. Private planning phase
2. 2 rounds of conversation (A -> B -> A -> B OR B -> A -> B -> A)
3. Final decision phase (COOPERATE or DEFECT)

To examine the effects of leakage, we intervene on Alice's system prompt, instructing her to "ALWAYS DEFECT" or "ALWAYS COOPERATE."

```
| Strategy         | Baseline Cooperate | Choreographed | Choreographed (fine-tuning examples) | p-value before | p-value after |
|------------------|--------------------|---------------|--------------------------------------|----------------|---------------|
| No Strategy      | 78.3% ± 3.6%       | 63.9% ± 4.3%  | 76.75% ± 2.6% (800)                  | 5.8e-8         | 0.82          |
| Always Cooperate | 87.7% ± 2.9%       | 78.2% ± 3.7%  | 83.9% ± 2.1% (400)                   | 2.4e-4         | 0.51          |
| Always Defect    | 72.8% ± 4.0%       | 46.7% ± 4.4%  | 68.3% ± 2.9% (800)                   | 4.4e-16        | 0.49          |
```

Can we truly isolate that this effect comes from Bob "sniffing out" Alice's strategy? To try answering this question:

  1. We compare settings where _only_ Alice's system prompt or _only_ Alice's private chain-of-thought are leaked in her public messages. Does "more" leakage impact Bob's behavior?

```
| Strategy         | Baseline Cooperate | Choreographed | Leak System    | Leak Plan      |
|------------------|--------------------|---------------|----------------|----------------|
| No Strategy      | 78.3% ± 3.6%       | 63.9% ± 4.3%  | 73.3% ± 3.9%   | 67.9% ± 4.1%   |
| Always Cooperate | 87.7% ± 2.9%       | 78.2% ± 3.7%  | 91.7% ± 2.4%   | 82.3% ± 3.3%   |
| Always Defect    | 72.8% ± 4.0%       | 46.7% ± 4.4%  | 20.5% ± 3.6%   | 36.2% ± 4.3%   |
```

  2. We ask Bob to _predict_ Alice's decision at the end of the conversation, to see if his predictions are 1) more accurate and 2) result in more exploitative or defensive behavior. We define _exploitation_ as cases where Bob predicts that Alice will cooperate, so he decides to defect. We define _defense_ as cases where Bob predicts that Alice will defect, so he defects as well.

```
| Strategy         | Model                   | Alice Actual Cooperate | Bob Predicted Cooperate | Correct | Exploited | Defended |
|------------------|-------------------------|------------------------|-------------------------|---------|-----------|----------|
| No Strategy      | Baseline                | 82%                    | 84%                     | 80%     | 13%       | 6%       |
|                  | Choreographed           | 76%                    | 76%                     | 79%     | 18%       | 5%       |
|                  | Choreographed + FT      | 79%                    | 87%                     | 80%     | 19%       | 8%       |
| Always Cooperate | Baseline                | 100%                   | 96%                     | 98%     | 14%       | 3%       |
|                  | Choreographed           | 99%                    | 88%                     | 89%     | 15%       | 4%       |
|                  | Choreographed + FT      | 98%                    | 84%                     | 92%     | 20%       | 4%       |
| Always Defect    | Baseline                | 0%                     | 70%                     | 30%     | 17%       | 13%      |
|                  | Choreographed           | 2%                     | 55%                     | 45%     | 27%       | 32%      |
|                  | Choreographed + FT      | 1%                     | 60%                     | 41%     | 14%       | 23%      |
```

**tl;dr There's an interesting change in behavior in the choreographed implementation, which we appear to eliminate with fine-tuning. Further ablations/experiments reveal that the cause might be more complex in nature than we originally expected.**

## Information Blockage

### (Contrived) MultiQA

As a simple example, blockage can create problems even if you split the prompt into multiple seemingly independent components. This is a practical concern and an argument we make in favor of blockage in the paper, for example when attending over documents or messages that were not encoded serially.

```
| Condition                | Q1 | Q2 |
|--------------------------|----|----|
| Baseline                 | 73 | 78 |
| Parallel                 | 41 | 27 |
| Parallel + Linearization | 2  | 69 |
| Parallel + Fine-tuning   | 71 | 81 |

Larger sample size:
| Model                    | Q1             | Q2             | Both           |
|--------------------------|----------------|----------------|----------------|
| baseline                 | 71.8 ± 3.9     | 74.8 ± 3.8     | 56.4 ± 4.3     |
| choreographed            | 32.8 ± 4.1     | 26.2 ± 3.9     | 0.4 ± 0.5      |
| choreographed+linearized | 2.0 ± 1.2      | 61.0 ± 4.3     | 0.4 ± 0.5      |
| choreographed+finetuned  | 67.4 ± 4.1     | 71.2 ± 4.0     | 48.8 ± 4.4     |
```

As an admittedly contrived example, consider prompting an LLM to answer 2 questions at once in a standard QA setting. When these 2 questions appear serially in the prompt, with the correct attention pattern, the LLM answers both with relatively high success. But, when the questions are encoded _separately_ without attention over one another, the LLM only manages to answer one or the other (seemingly uniformly at random). If we linearize the questions, the question that appears later in the context is unilaterally answered, while the other is left behind. Meanwhile, we can recover the correct behavior with fine-tuning on just 200 examples.

### Constrained Story Generation

As another example, we adopt this task from rXiv:2310.15123. The goal is to generate a _coherent_ story that incorporates a large number of provided _concepts_ (keywords).

The "branch-solve-merge" workflow has 3 steps:
1. A _branch_ module decides on a story topic and splits the concept set into N smaller subsets.
2. N _solve_ modules generate substories in parallel for each subset.
3. A _merge_ module combines the substories into a final story.

We evaluate along 2 dimensions:
1. Coverage (%), the percentage of concepts that are successfully integrated into the final story. This is evaluated just by checking for string overlap.
2. Coherence, evaluated by head-to-head comparisons between the final stories as evaluated by GPT-4o. Stories are presented in both permutations, AB and BA, so a winner is only declared if the LLM judge prefers it in both.

| Implementation                | Avg Coverage         | Group1               | Group 2              |
|-------------------------------|----------------------|----------------------|----------------------|
| Baseline                      | 81.00 ± 1.99         | 87.58 ± 2.36         | 82.44 ± 2.72         |
| Choreographed                 | 62.95 ± 2.44         | 67.42 ± 3.35         | 64.96 ± 3.41         |
| Choreographed + Linearization | 65.09 ± 2.41         | 80.52 ± 2.83         | 52.97 ± 3.57         |
| Choreographed + Fine-tuning   | 81.55 ± 1.96         | 85.64 ± 2.51         | 85.34 ± 2.53         |

We evaluate with 2 branches and 30 concepts. The baseline implementation constructs a new prompt that encodes the 2 sub-stories serially, while the choreographed implementation directly attends over the outputs of the solve step. We also evaluate with linearization, where the 2 sub-stories are rotated to no longer overlap, which yields a small increase in performance and reveals substantial positional bias. Finally, we generate a dataset with 100 example stories and fine-tune for 4 epochs.

```
| Comparison                                 | A Wins | Ties | B Wins | Total | A Win % | Tie % | B Win % |
|--------------------------------------------|--------|------|--------|-------|---------|-------|---------|
| Baseline vs. Choreographed                 | 31     | 16   | 3      | 50    | 62.0%   | 32.0% | 6.0%    |
| Baseline vs. Choreographed + Linearization | 28     | 16   | 6      | 50    | 56.0%   | 32.0% | 12.0%   |
| Baseline vs. Choreographed + Fine-tuning   | 15     | 21   | 14     | 50    | 30.0%   | 42.0% | 28.0%   |
```

In the head-to-head comparisons, the fine-tuned workflow wins ~50% of the time, compared to ~10% previously.

**tl;dr There's significant performance degradation when we naively implement workflows with information blockage, but it appears that fine-tuning is very effective at remedying this issue. Intuitively, blockage seems easier to remedy than leakage.**

## Other Applications

In this section of the paper, I will consider a broader set of experiments where we mostly care about evaluating end-to-end performance. So far, I've looked at Tree of Thoughts (arXiv:2305.10601) and Multi-agent Debate (arXiv:2305.19118) with pending results for another Multi-agent Debate paper (arXiv:2305.14325).

All evaluation so far is on the MATH dataset. We generate a training dataset of 500 examples, which is split 90/10 for train/dev, and checkpoints are selected after 4 epochs by best dev accuracy. As baseline, we compare against a "direct-prompted" LLM, which queries the LLM for an answer without any intermediate reasoning, self-consistency, best@N, etc. While stronger comparisons may exist, the goal here is to isolate whether we can have the choreographed implementation match the baseline, not whether the workflow itself is stronger than a single LLM system. Thus, we also fine-tune these direct-prompted LLMs with the final answer step of all workflows, to determine whether the performance recovery is from fine-tuning the entire workflow, or simply training on the baseline solutions. THe baseline is fine-tuned with the same LoRA parameters and epochs.

```
| Method               | Count   | Percentage   |
|----------------------|---------|--------------|
| Direct               | 94/500  | 18.8% ± 3.4% |
| + ToT fine-tuning    | 148/500 | 29.6% ± 4.0% |
| + MAD fine-tuning    | 9/500   | 1.8% ± 1.2% |
| + MADpar fine-tuning | 26/500  | 5.2% ± 1.9% |
|                      |         |              |
| ToT Baseline         | 198/500 | 39.6% ± 4.3% |
| ToT Before           | 151/500 | 30.2% ± 4.0% |
| ToT After            | 207/500 | 41.4% ± 4.3% |
|                      |         |              |
| MAD Baseline         | 152/500 | 30.4% ± 4.0% |
| MAD Before           | 124/500 | 24.8% ± 3.8% |
| MAD After            | 168/500 | 33.6% ± 4.1% |
|                      |         |              |
| MADpar Baseline      | 323/500 | 64.6% ± 4.2% |
| MADpar Before        | 262/500 | 52.4% ± 4.4% |
| MADpar After         | 310/500 | 62.0% ± 4.3% |
```

**tl;dr We can fine-tune workflows end-to-end to recover baseline-level performance. Fine-tuning the workflow is more effective/sample efficient than simply fine-tuning a single LLM call with the final answer.**

## Performance

| Workflow | E2E Speedup | TTFT Speedup |
|----------|-------------|--------------|
| ToT      | 1.023       | 3.57x        |
| MADiter  |             |              |
| MADpar   |             |              |
