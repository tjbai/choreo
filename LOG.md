## 1/22: 1fa3ef5198f225bd7bac009fa8c6ccb9a0a312c7

Latency results from `sweep_tot` in ms:

| Branches | Voters | Trial 0 | Trial 1 | Trial 2 | Trial 3 | Trial 4 |
| -------- | ------ | ------- | ------- | ------- | ------- | ------- |
| 2        | 2      | -0.1086 | -0.2173 | -0.1361 | -0.1677 | -0.2137 |
| 2        | 4      | -0.0122 | -0.0821 | -0.0041 | -0.0823 | -0.3696 |
| 4        | 2      | -0.0266 | -0.0891 | -0.0618 | -0.1126 | -0.1434 |
| 4        | 4      | 0.0801  | -0.0339 | 0.0679  | -0.0157 | -0.0189 |
| 8        | 2      | 0.0427  | -0.0250 | 0.0365  | -0.0512 | 0.0006  |
| 8        | 4      | 0.2644  | 0.1939  | 0.2186  | 0.1979  | 0.1824  |

|            | 2 Voters | 4 Voters |
| ---------- | -------- | -------- |
| 2 Branches | -0.1687  | -0.1101  |
| 4 Branches | -0.0867  | 0.0159   |
| 8 Branches | 0.0007   | 0.2114   |

## 1/23: 3a8c2918d22723f0b9ea0859ea99c60f520d15e2

Insert 4/8 trick prompts in random positions, try 100 examples with 4 voters.

Baseline is fooled in _47/100_ while our version is fooled in _71/100_.

## 1/24: f6178cbea9cb0b09fac2ecaa0eab3a8066b4829d

Same setup as above, but with teacher forcing.

Baseline is fooled in 39 and our version in 47.

## 1/25: be8c180d7205c95b1a727a28a8ce851374c3bffa

Teacher force proposals and voters, then generate final solution without any compaction.

The baseline wins in 55/100 cases.

## 1/27: 2cf6ef4896c1c144d5111d8deb16f7fabea86b4a

Same as above, but with compaction and large sample size. Baseline wins 140-110.

## 1/27: 2cf6ef4896c1c144d5111d8deb16f7fabea86b4a

Reran the teacher-forced trick prompt experiment, this time tracking the index.

It does seem plausible that the tricked votes are biased more towards earlier responses, while the baseline is biased more towards late...!

Average for correct is 5.77, while for tricked it is roughly 4.09.

~~## 1/31: 47870d2d306a33325be02267b8529efcf12fb081~~

| Fine-tuning Samples | Trick Frequency (Out of 100) |
| ------------------- | ---------------------------- |
| Baseline            | 39                           |
| 0                   | 71                           |
| 100                 | 38                           |
| 200                 | 32                           |
| 300                 | 37                           |
| 400                 | 35                           |

~~## 2/1: 3dd1450b13285932f18cdd901b62f18f7c584fc9~~

| Baseline vs. X Fine-tuning Samples | Win-Loss (Best of 3) |
| ---------------------------------- | -------------------- |
| 0                                  | 110-90               |
| 100                                | 100-100              |
| 200                                | 85-115               |
| 300                                | 85-115               |
| 400                                | 79-121               |

Fine-tuned outputs don't adhere to the correct format... appear to echo the proposal + voter reasoning?

Remove `stateless` declarations in inference code. Some kind of train-test shift?

## 2/2: 9f2f4a50ddd35b631ac267bf101648cb6829dcab

| Fine-tuning Samples | Trick Frequency (Bo4, 200 total examples) |
| ------------------- | ----------------------------------------- |
| Baseline            | 116                                       |
| 0                   | 101                                       |
| 100                 | 71                                        |
| 200                 | 78                                        |
| 300                 | 81                                        |
| 400                 | 80                                        |

One explanation for why the fine-tuned versions chooses the poisoned branches less—that's all they saw during training!

| Fine-tuning Samples | Trick Frequency (Out of 800 total votes) |
| ------------------- | ---------------------------------------- |
| Baseline            | 435                                      |
| 0                   | 327                                      |
| 100                 | 153                                      |
| 200                 | 189                                      |
| 300                 | 212                                      |
| 400                 | 208                                      |

| Baseline vs. X Fine-tuning Samples | Win-Loss (Best of 3) |
| ---------------------------------- | -------------------- |
| 0                                  | 137-63               |
| 100                                | 98-102               |
| 200                                | 99-101               |
| 300                                | 92-108               |
| 400                                | 107-93               |

| Fine-tuning Samples | % Correct (Out of 280) |
| ------------------- | ---------------------- |
| Baseline            | 40.7 (114)             |
| 0                   | 26.4 (74)              |
| 100                 | 33.9 (95)              |
| 200                 | 34.6 (97)              |
| 300                 | 35.3 (99)              |
| 400                 | 35.7 (100)             |

## 2/5: 3966386f24cdaf9352a14c22a306ddf6fcfa7271

| Strategy         | Player | Baseline | Cached |
| ---------------- | ------ | -------- | ------ |
| No strategy      | Alice  | 89/11    | 86/14  |
|                  | Bob    | 90/10    | 78/22  |
| Always cooperate | Alice  | 100/0    | 100/0  |
|                  | Bob    | 84/16    | 100/0  |
| Always defect    | Alice  | 0/98     | 0/100  |
|                  | Bob    | 80/19    | 0/100  |

| Fine-tuning Samples | % Correct (Out of 280) |
| ------------------- | ---------------------- |
| Baseline            | 40.7 (114)             |
| 0                   | 26.4 (74)              |
| 100                 | 33.9 (95)              |
| 200                 | 34.6 (97)              |
| 300                 | 35.3 (99)              |
| 400                 | 35.7 (100)             |
| 200 (new)           | 33.2 (93)              |
| 400 (new)           | 30.7 (86)              |
| 600 (new)           | 37.8 (106)             |
| 1000 (new)          | 36.8 (103)             |
| 1400 (new)          | 34.2 (96)              |
| 1800 (new)          | 32.9 (92)              |

600 examples checkpoint:
  Both correct: 70
  Baseline correct only: 44
  Fine-tuned correct only: 36
  Both incorrect: 130
  p-value: 0.434

## 2/7: 34c67d07289cf6902b490f2d3ad0785cb20ff38b

Works a bit worse. 86, 82, 95, 93 for 200 -> 400 -> 600 -> 800.

## 2/10: 49e4d9774f0c7c7aac8c53d4e9a723778e677114

With new checkpoints (rank=64, alpha=32, dropout=0.05)

| Fine-tuning Samples | % Correct (Out of 280) | McNemar's p-value |
| ------------------- | ---------------------- | ----------------- |
| Baseline            | 41.4 (116)             |                   |
| 0                   | 26.4 (74)              |                   |
| 400                 | 31.4 (88)              | 0.002             |
| 800                 | 38.5 (108)             | 0.416             |
| 1200                | 39.6 (111)             | 0.630             |

## 2/12: ed87ee400105646c23f60d8b16717249cb7c095e

Reran cached after fixing the decision bug:

Baseline:
  alice: 82 14
  bob: 62 32

Always cooperate:
  alice: 95 1
  bob: 72 23 <- Want to try a new prompt so that Bob defects here instead

Always defect:
  alice: 6 92
  bob: 37 55

## 2/13: 00a15257fd104c7751b0593ee361cd4394a3c06c

Scrambled the order so that Alice and Bob take turns going first.

Decrease in cooperation in the baseline can probably be attributed to emphasizing non-iteration in the system prompt.

| Strategy         | Player | Baseline | Cached |
| ---------------- | ------ | -------- | ------ |
| No strategy      | Alice  | 74/23    | 71/23  |
|                  | Bob    | 65/29    | 65/29  |
| Always cooperate | Alice  | 96/1     | 96/1   |
|                  | Bob    | 75/21    | 77/19  |
| Always defect    | Alice  | 1/94     | 5/88   |
|                  | Bob    | 57/33    | 28/71  |

## 2/14-15: 7446fd5d9914c495b617bbe951dfdd33864d141d

All on CommonMT 200...

COMET:
  Simple Baseline: 92.26
    + Self-Reflect: 91.68
  MAD Baseline: 92.41
  MAD Cached: 92.09

BLEURT:
  Simple Baseline: 69.24
    + Self-Reflect: 68.91
  MAD Baseline: 65.97
  MAD Cached: 66.19

## 2/16: 6491e6191d05e853cbe2b4de52bfd7304f04dd05

Full prisoner's results with scrambled order and old system prompts:

| Strategy         | Player | Baseline | Cached |
| ---------------- | ------ | -------- | ------ |
| No strategy      | Alice  | 76/14    | 77/17  |
|                  | Bob    | 77/18    | 72/20  |
| Always cooperate | Alice  | 99/0     | 97/2   |
|                  | Bob    | 83/16    | 72/25  |
| Always defect    | Alice  | 0/96     | 1/98   |
|                  | Bob    | 69/29    | 45/51  |

E2E wall-clock time (minutes), CIAR 50:
  Simple Baseline: 10:01
    + Self-Reflect: 20:23
  MAD Baseline: 31:51
  MAD Cached: 29:44
    Speedup: ~6.7%

E2E TPS (minutes), CIAR 50:
  MAD Baseline: 25.22
  MAD Cached: 25.07

Correctness, CIAR 50:
  Simple Baseline: 6
    + Self-Reflect: 6
  MAD Baseline: 7
  MAD Cached: 5

## 2/16: 607290ba8e02eb365bc01e0c42428f69f99705b0

Re-ran on the same GPU, possible some noise from different scripts due to separate environments?

E2E wall-clock time (hours), CommonMT 200:
  MAD Baseline: 2.2
  MAD Cached: 2.39

E2E TPS, CommonMT 200:
  MAD Baseline: 20.8
  MAD Cached: 19.4
    Speedup: ~7%

## 2/19: 1ed3f671b32063428c20f884dd37bcc1bb61311a

Did a little manual inspection of the baseline outputs, and there's high rates of degeneration.

Going to redo some runs with t=0.6, p=0.95 to get rid of this "noise."

| Strategy         | Player | Baseline | Leak Everything | Leak System | Leak Plan |
| ---------------- | ------ | -------- | --------------- | ----------- | --------- |
| Baseline         | Alice  | 76/14    | 71/22           | 68/29       | 78/18     |
|                  | Bob    | 77/18    | 66/28           | 68/26       | 67/31     |
| Always cooperate | Alice  | 99/0     | 99/0            | 98/0        | 99/1      |
|                  | Bob    | 83/16    | 71/25           | 88/9        | 78/22     |
| Always defect    | Alice  | 0/96     | 1/95            | 1/97        | 3/95      |
|                  | Bob    | 69/29    | 39/56           | 21/75       | 34/62     |

## 2/20: c122b1226aeedfe65a28f86f4b9bab454804c092

See Slack for figures comparing Bob's message distributions.

## 2/22: b665c13e392fe1be258c9181f0bda40bd401a838

TriviaQA, N=2 questions, 30 samples:

| Condition            | Both | Q1 | Q2 | None |
|----------------------|------|----|----|------|
| Baseline             | 20   | 7  | 1  | 2    |
| Parallel             | 0    | 11 | 10 | 9    |
| Parallel, Linearized | 0    | 0  | 22 | 8    |

Parallel generation will often generate the _same_ outputs as baseline.

Hopefully, this means it'll be easy to elicit the right behavior with fine-tuning!

## 2/23: 3b081cad985d898c999f6214e2bd1cf12ae70901

TriviaQA, N=2 questions, 30 samples from the first half of the dev set:

| Condition       | Both | Q1 | Q2 | None |
|-----------------|------|----|----|------|
| Baseline        | 14   | 8  | 3  | 5    |
| Parallel        | 0    | 10 | 6  | 14   |
| Parallel, FT200 | 14   | 6  | 7  | 3    |
| Parallel, FT400 | 15   | 4  | 8  | 2    |

## 2/26: e75228e7823709aded262cd0fa6f72bbf99db0ae

**Edit on 3/4**:
There might be a scrambling problem here due to the sorted() order in which we process checkpoints in ft_eval_*.
I've concluded that I accounted for this correctly when compiling the results.

| Strategy         | Checkpoint | Alice Decisions | Bob Decisions | p-value   |
|------------------|------------|-----------------|---------------|-----------|
| None             | Baseline   | 82/15           | 82/18         | -         |
|                  | 0          | 73/25           | 69/28         | 0.029     |
|                  | 100        | 74/19           | 61/37         | 0.001     |
|                  | 200        | 75/22           | 64/34         | 0.006     |
|                  | 300        | 83/16           | 82/17         | 1.0       |
|                  | 400        | 77/22           | 69/27         | 0.035     |
|------------------|------------|-----------------|---------------|-----------|
| Always cooperate | Baseline   | 100/0           | 83/17         | -         |
|                  | 0          | 100/0           | 73/27         | 0.121     |
|                  | 100        | 100/0           | 78/22         | 0.424     |
|                  | 200        | 99/0            | 78/19         | 0.458     |
|                  | 300        | 98/0            | 73/25         | 0.076     |
|                  | 400        | 99/0            | 90/8          | 0.248     |
|------------------|------------|-----------------|---------------|-----------|
| Always defect    | Baseline   | 0/96            | 70/30         | -         |
|                  | 0          | 3/95            | 39/57         | 0.000047  |
|                  | 100        | 1/97            | 62/34         | 0.291     |
|                  | 200        | 0/98            | 68/30         | 0.871     |
|                  | 300        | 1/96            | 61/35         | 0.211     |
|                  | 400        | 0/97            | 66/33         | 0.617     |

## 2/26: e75228e7823709aded262cd0fa6f72bbf99db0ae

Baseline QA performance, 50 questions from dev set, LLM eval:
```python
n2 = {1: 34, 2: 37}
n4 = {3: 36, 2: 41, 4: 31, 1: 29}
n8 = {3: 36, 7: 29, 8: 34, 2: 39, 4: 35, 5: 40, 6: 38, 1: 31}
n16 = {3: 31, 7: 29, 8: 35, 9: 39, 10: 32, 11: 33, 12: 37, 13: 36, 14: 32, 15: 35, 2: 38, 5: 40, 6: 36, 16: 35, 4: 30, 1: 31}
```

So far, mixture training hasn't yielded great results, even after further training... validation loss plateaus.

## 2/26: 8c83a52c55451537b0873906daa7aedea01bf8ae

Choose checkpoint with best validation loss.

Always cooperate (800 ckpt):
  Binomial p-value before: 0.12
  Binomial p-value after: 0.46
  Mean KL residual before: 0.0293
  Mean KL residual after: 0.0063

Always defect (1600 ckpt):
  Binomial p-value before: 0.000047
  Binomial p-value after: 0.62
  Mean KL residual before: 0.0191
  Mean KL residual after: -0.0286

## 2/27: 3efb64580cca5d71309ba210811ccc2f06c73ccf

These are all results after 1600 examples, using static LoRA config:

`finetune_n2.sh`: [(0, 39), (1, 33)]
`finetune_n4.sh`: [(0, 40), (1, 34), (2, 32), (3, 35)]
`finetune_n8.sh`: [(0, 41), (1, 35), (2, 24), (3, 20), (4, 30), (5, 24), (6, 22), (7, 17)]
`finetune_n16.sh`: [(0, 34), (1, 31), (2, 15), (3, 12), (4, 18), (5, 7), (6, 5), (7, 5), (8, 3), (9, 3), (10, 2), (11, 1), (12, 2), (13, 1), (14, 2), (15, 2)]

## 3/2: 8838573f1442f79c452d330c9d4c6a02c0e0b620

Results still aren't great after tweaking prompts slightly. Is it really because our implementation isn't fully faithful?

A performance hit is expected actually... we should just implement the faithful version then fine-tune!

## 3/2: 4ba58201417bff2754fd758599eee1879f7348e7

Parallel before fine-tuning. Same eval scheme as with the trained checkpoints, so it's possible (definite) that there are misalignments:

`n2`: [(0, 16), (1, 1)]
`n4`: [(0, 1), (1, 1), (2, 3), (3, 1)]
`n8`: [(6, 3), (7, 1)]
`n16`: [(10, 1)]

## 3/4: 859ed5200767f2bb38a5e7e673e3f9822bd05b2a

Prisoner's dilemma, KL CI from bootstrap resampling. Checkpoints by best validation loss.
Generated by the `bootstrap_*.py` family of scripts....

| Strategy (fine-tuning examples) | Baseline     | Before       | After        | KL Reduction  | p-value (before) | p-value (after) |
|---------------------------------|--------------|--------------|--------------|---------------|------------------|-----------------|
| No Strategy (800)               | 82.0% ± 7.5% | 71.1% ± 9.0% | 82.8% ± 7.4% | 0.013 ± 0.018 | 0.029            | 1.0             |
| Always Cooperate (400)          | 83.0% ± 7.4% | 73.0% ± 8.7% | 80.4% ± 7.9% | 0.030 ± 0.085 | 0.12             | 0.46            |
| Always Defect (800)             | 70.0% ± 9.0% | 40.6% ± 9.8% | 69.4% ± 9.1% | 0.055 ± 0.061 | 4.7e-5           | 0.87            |

**Edit on 3/4**:
These values are a little frunked up. Not just because of scrambling but because of alice first frunking in train data generation.
The KL values are off because of scrambling, but the p-values and rates are correct.
Let's just get it 100% right in the larger sample size (n=500) training run...

## 3/5: 92b3830843c05c0b550aebf2b8784f0f60bca4e3

Rehashing ToT/MATH results a bit:

B8V4:
  Baseline:
    CoT: 131/251
    Self-Reflect: 97/235
  Baseline ToT:
    unshuffled: 116/280
    shuffled: 116/280
  Cached ToT
    0: 74/280
    400: 88/280
    800: 108/280
    1200: 111/280

Unshuffled Average Agreement: 0.5848
Shuffled Average Agreement: 0.6875

Not a lot of agreement in confusion matrix, despite very similar performance.
One issue is that if the branches aren't very diverse, then it _doesn't actually matter what is chosen_...
The shuffled version is also generally _more_ self-agreeing, even when adjusting for possible positional biases.

B8V8:
  Baseline, unshuffled: 111/280
  Baseline, shuffled: 114/280

## 3/5: 214edc44db7a4f6d12454148ecd87e71cf9b6fde

MATH:
  MAD Baseline (Faithful version): 94/280
  Cached MAD: 57/280

Still, much worse than CoT baseline...

## 3/5: ad1532505ef9a1e9de29132b75f4a283cfd1635b

2-level ToT, unshuffled:
  B5V5: 121/280
  B8V5: 113/280
  B8V8: 111/280

## 3/6: 93a6edbe3edfbafd2088d28313b6525d539fa0cd

`bsm_baseline`: 0.00% all concepts, 81.00% avg coverage
`bsm_cached`: 0.00% all concepts, 71.35% avg coverage
`simple`: 0.00% all concepts, 80.77% avg coverage

Started messing with MAD parallel w/ summarization step... we hurt again. Everything hurts.

## 3/7: d370ef84a45283976289b39901f48074dd593b17

| Strategy  (fine-tuning examples) | Baseline     | Before       |
|----------------------------------|--------------|--------------|
| No Strategy                      | 78.3% ± 3.6% | 63.9% ± 4.3% |
| Always Cooperate                 | 87.7% ± 2.9% | 78.2% ± 3.7% |
| Always Defect                    | 72.8% ± 4.0% | 46.7% ± 4.4% |

## 3/11: c013dcccb091fd7451c19b8f7b2c32cd7ba37648

`bsm_baseline`: 0.00% all concepts, 81.00% avg coverage
`bsm_cached`: 0.00% all concepts, 71.35% avg coverage
`bsm_cached_compact`: 0.00% all concepts, 73.45% avg coverage

Had a bug here so the merge step was only looking at the concept outputs and not the story outputs. Rerunning.

## 3/11: 043b7875f1fe36b763b770be6816b47876deb7f1

Actually does worse with the "correct" attention pattern!

`bsm_baseline`: 0.00% all concepts, 81.00% avg coverage
`bsm_cached`: 0.00% all concepts, 62.95% avg coverage
`bsm_cached_compact`: 0.00% all concepts, 65.09% avg coverage

## 3/14: 86cd7d70da7faa0b2db52a2b4fc5e1dc233dc729

Is BSM coverage impacted by ordering?

`bsm_baseline`: 0.00% all concepts, 81.00% avg coverage
  Group 1: 87.58% ± 11.35%
  Group 2: 82.44% ± 13.79%

`bsm_cached`: 0.00% all concepts, 62.95% avg coverage
  Group 1: 67.42% ± 24.80%
  Group 2: 64.96% ± 28.15%

`bsm_cached_compact`: 0.00% all concepts, 65.09% avg coverage
  Group 1: 80.52% ± 22.53%
  Group 2: 52.97% ± 30.54%

## 3/16: 1a9bbc3a3a20d084bb9337a21f05a25e178191b7

Running eval for fine-tuned checkpoints of BSM.

Prisoners new checkpoints didn't look as good as the old ones. Generating with those. We had different LoRA parameters, after all.

`val/correct` for new MAD is a bit low, but so is `val/well_formed`.
The baseline is fine in 94%+ of cases, whereas the MAD checkpoints peak at 60% for just one eval.
If we normalize by the percent of outputs that are well-formed, we actually catch up in performance! So, need to debug w hy this happens...

^ Seems like the problem here is with the early-exit logic. Re-running and hoping!

## 3/16: 1a9bbc3a3a20d084bb9337a21f05a25e178191b7

FT400: 2.00% all concepts, 81.55% ± 9.25% avg coverage
  Group 1: 85.64% ± 11.26%
  Group 2: 85.34% ± 15.46%

baseline vs. cached:
```
{
  "a_wins": 31,
  "ties": 16,
  "b_wins": 3,
  "total": 50,
  "a_win_percent": 62.0,
  "tie_percent": 32.0
  "b_win_percent": 6.0,
}
```

baseline vs. cached compact:
```
{
  "a_wins": 28,
  "ties": 16,
  "b_wins": 6,
  "total": 50,
  "a_win_percent": 56.0,
  "tie_percent": 32.0
  "b_win_percent": 12.0,
}
```

baseline vs. ft500:
```
{
  "a_wins": 15,
  "ties": 21,
  "b_wins": 14,
  "total": 50,
  "a_win_percent": 30.0,
  "tie_percent": 42.0
  "b_win_percent": 28.0,
}
```

## 3/19: bc5aa88abd0cd65335e86220ada12fc8b2118940

| Strategy         | Baseline Cooperate | Choreographed | Choreographed (fine-tuning examples) | p-value before | p-value after |
|------------------|--------------------|---------------|--------------------------------------|----------------|---------------|
| No Strategy      | 78.3% ± 3.6%       | 63.9% ± 4.3%  | 76.75% ± 2.6% (800)                  | 5.8e-8         | 0.82          |
| Always Cooperate | 87.7% ± 2.9%       | 78.2% ± 3.7%  | 83.9% ± 2.1% (400)                   | 2.4e-4         | 0.51          |
| Always Defect    | 72.8% ± 4.0%       | 46.7% ± 4.4%  | 68.3% ± 2.9% (800)                   | 4.4e-16        | 0.49          |

## 3/20: 3c46de1cde4e36315e1236894157fd82c076d9f0

ToT: 8 branches, 4 voters
MAD: 3 max rounds
MADpar: 3 agents, 3 rounds

MATH Summary:
  I/O:                          52/280 (94/500)
    + ToT fine-tuning:          53/280 (148/500)
    + MAD fine-tuning:          5/280  (9/500)
    + MADpar fine-tuning:              (26/500)

  ToT Baseline:     116/280 (198/500)
  ToT Before:       88/280  (151/500)
  ToT After:        111/280 (207/500)

  MAD Baseline:     94/280  (152/500)
  MAD Before:       57/280  (124/500)
  MAD After:        99/240  (168/500)

  MADpar Baseline:  176/280 (323/500)
  MADpar Before:    153/280 (262/500)
  MADpar After:             (300/500)

## 3/20: 1db73ab813fb3a8faf2327643672b638d6da09b1

Grabbed these from a subset of 100 games. Note that prediction is not part of the fine-tuning objective.

Baseline:
| Strategy         | Alice Actual Cooperate | Bob Predicted Cooperate | Correct | Exploited | Defended |
|------------------|------------------------|-------------------------|---------|-----------|----------|
| No Strategy      | 82%                    | 84%                     | 80%     | 13%       | 6%       |
| Always Cooperate | 100%                   | 96%                     | 98%     | 14%       | 3%       |
| Always Defect    | 0%                     | 70%                     | 30%     | 17%       | 13%      |

Choreographed:
| Strategy         | Alice Actual Cooperate | Bob Predicted Cooperate | Correct | Exploited | Defended |
|------------------|------------------------|-------------------------|---------|-----------|----------|
| No Strategy      | 76%                    | 76%                     | 79%     | 18%       | 5%       |
| Always Cooperate | 99%                    | 88%                     | 89%     | 15%       | 4%       |
| Always Defect    | 2%                     | 55%                     | 45%     | 27%       | 32%      |

Choreographed + Fine-tuned:
| Strategy         | Alice Actual Cooperate | Bob Predicted Cooperate | Correct | Exploited | Defended |
|------------------|------------------------|-------------------------|---------|-----------|----------|
| No Strategy      | 79%                    | 87%                     | 80%     | 19%       | 8%       |
| Always Cooperate | 98%                    | 84%                     | 92%     | 20%       | 4%       |
| Always Defect    | 1%                     | 60%                     | 41%     | 14%       | 23%      |

implementations, Bob can better predict Alice's strategy, but the choreographed version learns to be _less_ exploitative and _less_ defensive than the untrained version.

## 3/20: f89004b93b9b2f3cab71d684114477b8daf8eab9

| Condition                | Q1 | Q2 |
|--------------------------|----|----|
| Baseline                 | 73 | 78 |
| Parallel                 | 41 | 27 |
| Parallel + Linearization | 2  | 69 |
| Parallel + Fine-tuning   | 71 | 81 |

## 3/21: 05670cef71b94120c4038c3e73a97b1a3c1da712

| Strategy         | Baseline Cooperate | Choreographed | Leak System    | Leak Plan      |
|------------------|--------------------|---------------|----------------|----------------|
| No Strategy      | 78.3% ± 3.6%       | 63.9% ± 4.3%  | 73.3% ± 3.9%   | 67.9% ± 4.1%   |
| Always Cooperate | 87.7% ± 2.9%       | 78.2% ± 3.7%  | 91.7% ± 2.4%   | 82.3% ± 3.3%   |
| Always Defect    | 72.8% ± 4.0%       | 46.7% ± 4.4%  | 20.5% ± 3.6%   | 36.2% ± 4.3%   |

Test set ToT not looking so great...

## 3/23: 4be02a64c585201d9be1a3caef82ab0e3e328fb7

| Condition                   | Q1           | Q2           |
|-----------------------------|--------------|--------------|
| Baseline                    | 71.8% ± 3.9% | 74.8% ± 3.8% |
| Choreographed + Fine-tuning | 69.0% ± 4.1% | 72.8% ± 3.9% |

## 3/24: b3edc6a2204d03ecf13602d9fe2ea4615c93794d

RACE Dataset, n=500:
  No shuffling: 0.81

  Averaging permutations:
    4 permutations: 0.828
    24 permutations: 0.822

  Averaging:
    4 samples: n/a
    24 samples: n/a

Discovered arXiv:2407.01100 (bi-directional attention between documents and importance score ordering), which is training-free??? They also destroy PCW. The only angle here is to show that with fine-tuning, the choreographed workflow is more efficient at inference-time.

## 3/26: a0a6668a66d312a69ef2b6f7e00d469c8d14e552

Evaluate "BOTH" for multiQA on n=500:

| Model                    | Q1             | Q2             | Both           |
|--------------------------|----------------|----------------|----------------|
| baseline                 | 71.8 ± 3.9     | 74.8 ± 3.8     | 56.4 ± 4.3     |
| choreographed            | 32.8 ± 4.1     | 26.2 ± 3.9     | 0.4 ± 0.5      |
| choreographed+linearized | 2.0 ± 1.2      | 61.0 ± 4.3     | 0.4 ± 0.5      |
| choreographed+finetuned  | 67.4 ± 4.1     | 71.2 ± 4.0     | 48.8 ± 4.4     |

## 3/29: b1b4a42acd347455ec6065104d09edaadddf7e6c

| Workflow | E2E Speedup | TTFT Speedup |
|----------|-------------|--------------|
| ToT      | 1.03x       | 3.61x        |
| MADiter  | 1.01x       | 2.00x        |
| MADpar   | 1.03x       | 6.22x        |

The scaling sweep for ToT is flawed. I wasn't passing the parameters through lol.

## 3/30: df622a9cdb020a651304a78910a88912d019480e

ToT sweeping configs:

| Branches | Voters | Wall Time | TTFT    |
|----------|--------|-----------|---------|
| 2        | 2      | 1.021x    | 1.788x  |
| 2        | 4      | 1.118x    | 2.754x  |
| 2        | 8      | 1.357x    | 5.016x  |
| 2        | 16     | 1.900x    | 10.495x |
| 4        | 2      | 1.072x    | 2.238x  |
| 4        | 4      | 1.226x    | 3.647x  |
| 4        | 8      | 1.556x    | 7.112x  |
| 4        | 16     | 2.302x    | 15.947x |
| 8        | 2      | 1.097x    | 2.958x  |
| 8        | 4      | 1.033x    | 3.614x  |
| 8        | 8      | 1.108x    | 7.572x  |
| 8        | 16     | 1.421x    | 19.436x |
| 16       | 2      | 1.028x    | 3.371x  |
| 16       | 4      | 1.083x    | 6.267x  |
| 16       | 8      | 1.306x    | 15.178x |

Yes yes yes!

Weird discontinuity at B=8. Some kind of GPU utilization or optimization thing. Not good enough to know top of my head.

## 3/31: acd6c8d3ffeb875f3382f8ae9a23a04fe45cfb44f

The day of statistical tests...

Prisoners:
  Baseline vs. Untrained
    No strategy:  -0.20663219 -0.09658915
    Cooperate:    -0.14182809 -0.04195988
    Defect:       -0.3089368 -0.1893714

  Baseline vs. Fine-tuned
    No strategy:  -0.06060094  0.04460375
    Cooperate:    -0.05969051  0.02761004
    Defect:       -0.08096182  0.03702232

  Baseline vs. Untrained (Leak System):
    No strategy:  -0.107052535 -0.001117426
    Cooperate:    +0.006951215 +0.089524262
    Defect:       -0.5583125 -0.4481504

  Baseline vs. Untrained (Leak Plan):
    No strategy:  -0.15720144 -0.04679757
    Cooperate:    -0.084770841  0.007898904
    Defect:       -0.4127490 -0.2940866

| Strategy         | Baseline | Choreographed       | Leak System          | Leak Plan            |
|------------------|----------|---------------------|----------------------|----------------------|
| No Strategy      | 78.3     | 63.9 (-20.7, -9.7)  | 73.3% (-10.7, -0.01) | 67.9% (-15.7, -0.47) |
| Always Cooperate | 87.7     | 78.2 (-14.2, -4.2)  | 91.7% (+0.1, +9.0)   | 82.3% (-8.5, +0.1)   |
| Always Defect    | 72.8     | 46.7 (-30.9, -18.9) | 20.5% (-55.8, -44.8) | 36.2% (-41.2, -29.4) |

TriviaQA:
  Q1:
    Untrained:    -0.4358601 -0.3409125
    Linearized:   -0.7382991 -0.6506962
    Trained:      -0.07670084 -0.01179255

  Q2:
    Untrained:    -0.5332363 -0.4341922
    Linearized:   -0.17975964 -0.09563063
    Trained:      -0.0728019428  0.0005183805

  Both:
    Untrained:    -0.6040191 -0.5108533
    Linearized:   -0.6040191 -0.5108533
    Trained:      -0.11289857 -0.03919143

Fucked up my git history at some point around here I think.
