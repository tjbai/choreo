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

