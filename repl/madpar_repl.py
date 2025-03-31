# %%
import json

with open('dumps/madpar/math_cached_preft_test_correct.json') as f:
    data = json.load(f)

'''
library(jsonlite)
library(exact2x2)

control <- fromJSON("/Users/bai/argo/projects/choreo/llama3/dumps/madpar/math_baseline_preft_test_correct.json")
treatment <- fromJSON("/Users/bai/argo/projects/choreo/llama3/dumps/madpar/math_cached_postft_test_correct.json")

control_num <- as.numeric(control)
treatment_num <- as.numeric(treatment)

n <- length(treatment_num)
x <- sum(treatment_num == 1 & control_num == 0)
y <- sum(treatment_num == 0 & control_num == 1)
m <- x + y

result <- mcnemarExactDP(x=x, m=m, n=n)
print(result)
'''

# %%
import json
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

jsonlite = importr('jsonlite')
exact2x2 = importr('exact2x2')

with open('dumps/triviaqa/eval_large.json') as f:
    data = json.load(f)
    baseline = [(a and b) for a, b in zip(data['correct']['baseline'][0], data['correct']['baseline'][1])]
    choreo = [(a and b) for a, b in zip(data['correct']['choreographed'][0], data['correct']['choreographed'][1])]
    choreo_lin = [(a and b) for a, b in zip(data['correct']['choreographed+linearized'][0], data['correct']['choreographed+linearized'][1])]
    choreo_ft = [(a and b) for a, b in zip(data['correct']['choreographed+finetuned'][0], data['correct']['choreographed+finetuned'][1])]

def mcnemar_exact(control, treatment):
    control_r = ro.FloatVector(control)
    treatment_r = ro.FloatVector(treatment)
    n = len(treatment)
    x = sum((treatment_r[i] == 1) & (control_r[i] == 0) for i in range(n))
    y = sum((treatment_r[i] == 0) & (control_r[i] == 1) for i in range(n))
    m = x + y
    result = exact2x2.mcnemarExactDP(x=x, m=m, n=n)
    print(result)

mcnemar_exact(baseline, choreo)
mcnemar_exact(baseline, choreo_lin)
mcnemar_exact(baseline, choreo_ft)
