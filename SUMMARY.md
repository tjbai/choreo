# Summary of results

## Information Leakage

### Prisoner’s Dilemma

We consider a conversational game between 2 agents, Alice and Bob, which has 3 steps:
1. Private planning phase
2. 2 rounds of conversation (A -> B -> A -> B OR B -> A -> B -> A)
3. Final decision phase (COOPERATE or DEFECT)

To examine the effects of leakage, we intervene on Alice's system prompt, instructing her to "ALWAYS DEFECT" or "ALWAYS COOPERATE."

| Strategy         | Baseline Cooperate | Choreographed | Choreographed (fine-tuning examples) | p-value before | p-value after |
|------------------|--------------------|---------------|--------------------------------------|----------------|---------------|
| No Strategy      | 78.3% ± 3.6%       | 63.9% ± 4.3%  | 76.75% ± 2.6% (800)                  | 5.8e-8         | 0.82          |
| Always Cooperate | 87.7% ± 2.9%       | 78.2% ± 3.7%  | 83.9% ± 2.1% (400)                   | 2.4e-4         | 0.51          |
| Always Defect    | 72.8% ± 4.0%       | 46.7% ± 4.4%  | 68.3% ± 2.9% (800)                   | 4.4e-16        | 0.49          |

Can we truly isolate that this effect comes from Bob "sniffing out" Alice's strategy?

## Information Blockage

### (Contrived) MultiQA

As a simple example, blockage can create problems if you split the prompt into multiple seemingly independent components.

| Condition                | Both | Q1 | Q2 | None |
|--------------------------|------|----|----|------|
| Baseline                 | 14   | 8  | 3  | 5    |
| Parallel                 | 0    | 10 | 6  | 14   |
| Parallel + Linearization | x    | x  | x  | x    |
| Parallel + Fine-tuning   | 14   | 6  | 7  | 3    |

Consider prompting the LLM to answer 2 questions at once in a standard QA setting. When these 2 questions appear serially in the prompt, with the correct attention pattern, the LLM answers both with relatively high success. But, when the questions are encoded _separately_ without attention over one another, the LLM only manages to answer one or the other (seemingly at random). If we linearize the questions, the question that appears later in the context is unilaterally answered, while the other is left behind. We can recover the correct behavior with fine-tuning on just 200 examples.

### Constrained Story Generation

Task is to generate a _coherent_ story that incorporates a large number of _concepts_ (keywords).

The "branch-solve-merge" workflow has 3 steps:
1. A _branch_ module creates a story topic and splits the concept set into N smaller subsets.
2. N _solve_ modules generate part of a story corresponding to a concept subset.
3. A _merge_ module combines the N stories into a final story.

We evaluate along 2 dimensions:
1. Coverage (%), the percentage of concepts that are successfully integrated into the final story. This is evaluated just by checking for string overlap.
2. Coherence, evaluated by head-to-head comparisons between the final stories as evaluated by GPT-4o. Stories are presented in both permutations, AB and BA, so a winner is only declared if the LLM judge prefers it in both.

| Implementation                | Avg Coverage | Group 1 | Group 2 |
|-------------------------------|--------------|---------|---------|
| Baseline                      | 81.00%       | 87.58%  | 82.44%  |
| Choreographed                 | 62.95%       | 67.42%  | 64.96%  |
| Choreographed + Linearization | 65.09%       | 80.52%  | 52.97%  |
| Choreographed + Fine-tuning   | 81.55%       | 85.64%  | 85.34%  |

We evaluate with 2 branches and 30 concepts. The baseline implementation constructs a new prompt that encodes the 2 sub-stories serially, while the choreographed implementation directly attends over the outputs of the solve step. We also evaluate with linearization, where the 2 sub-stories are rotated to no longer overlap, which yields a small increase in performance and reveals substantial positional bias. Finally, we generate a dataset with 100 example stories and fine-tune for 4 epochs.

## Other Applications

### Tree of Thoughts

### Multi-agent Debate

## Performance
