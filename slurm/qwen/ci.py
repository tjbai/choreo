import fire, json, re
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

exact2x2 = importr('exact2x2')
def mcnemar_exact(control, treatment):
    control_r = ro.FloatVector(control)
    treatment_r = ro.FloatVector(treatment)
    n = len(treatment)
    x = sum((treatment_r[i] == 1) & (control_r[i] == 0) for i in range(n))
    y = sum((treatment_r[i] == 0) & (control_r[i] == 1) for i in range(n))
    m = x + y
    result = exact2x2.mcnemarExactDP(x=x, m=m, n=n)

    result_str = str(result)
    match = re.search(r'95 percent confidence interval:\s+([-\d.]+)\s+([-\d.]+)', result_str)
    lower, upper = float(match.group(1)), float(match.group(2))
    return (round(100 * lower, 1), round(100 * upper, 1))

def main(a, b):
    with open(a) as f:
        a = json.load(f)

    with open(b) as f:
        b = json.load(f)

    a_corr = [d['judgment']['test_correct'] for d in a]
    b_corr = [d['judgment']['test_correct'] for d in b]
    N = min(len(a_corr), len(b_corr))
    print((round(100 * sum(a_corr) / N, 1), round(100 * sum(b_corr) / N, 1)))
    print(mcnemar_exact(a_corr[:N], b_corr[:N]))

if __name__ == '__main__':
    fire.Fire(main)
