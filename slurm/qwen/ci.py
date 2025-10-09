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
    print(result)

mcnemar_exact([0, 1, 1], [1, 0, 0])
