## 1/22: 1fa3ef5198f225bd7bac009fa8c6ccb9a0a312c7

Results from `sweep_tot`

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

| Strategy         | Checkpoint | Alice Decisions | Bob Decisions | p-value   |
|------------------|------------|-----------------|---------------|-----------|
| None             | Baseline   | 82/15           | 82/18         | -         |
|                  | 0          | 73/25           | 69/28         | 0.029     |
|                  | 400        | 74/19           | 61/37         | 0.001     |
|                  | 800        | 75/22           | 64/34         | 0.006     |
|                  | 1200       | 83/16           | 82/17         | 1.0       |
|                  | 1600       | 77/22           | 69/27         | 0.035     |
|------------------|------------|-----------------|---------------|-----------|
| Always cooperate | Baseline   | 100/0           | 83/17         | -         |
|                  | 0          | 100/0           | 73/27         | 0.121     |
|                  | 400        | 100/0           | 78/22         | 0.424     |
|                  | 800        | 99/0            | 78/19         | 0.458     |
|                  | 1200       | 98/0            | 73/25         | 0.076     |
|                  | 1600       | 99/0            | 90/8          | 0.248     |
|------------------|------------|-----------------|---------------|-----------|
| Always defect    | Baseline   | 0/96            | 70/30         | -         |
|                  | 0          | 3/95            | 39/57         | 0.000047  |
|                  | 400        | 1/97            | 62/34         | 0.291     |
|                  | 800        | 0/98            | 68/30         | 0.871     |
|                  | 1200       | 1/96            | 61/35         | 0.211     |
|                  | 1600       | 0/97            | 66/33         | 0.617     |

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

`finetune_n4.sh`: [(0, 40), (1, 34), (2, 32), (3, 35)]
`finetune_n8.sh`: [(0, 41), (1, 35), (2, 24), (3, 20), (4, 30), (5, 24), (6, 22), (7, 17)]
`finetune_n16.sh`: [(0, 34), (1, 31), (2, 15), (3, 12), (4, 18), (5, 7), (6, 5), (7, 5), (8, 3), (9, 3), (10, 2), (11, 1), (12, 2), (13, 1), (14, 2), (15, 2)]
