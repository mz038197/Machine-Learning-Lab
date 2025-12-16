import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)

# --- ex1data1: univariate (population -> profit), 97 rows ---
# keep a distribution similar to the classic dataset: many medium cities, few large cities
m1 = 97
mix = rng.uniform(0, 1, size=m1)
x1 = np.where(
    mix < 0.85,
    rng.uniform(4.5, 10.0, size=m1),   # most cities
    rng.uniform(10.0, 22.5, size=m1),  # a few larger cities
)  # population in 10,000s
w1_true, b1_true = 1.18, -3.65
noise1 = rng.normal(0, 1.6, size=m1)
y1 = w1_true * x1 + b1_true + noise1

# sort by x for nicer plots
order = np.argsort(x1)
x1, y1 = x1[order], y1[order]

ex1 = np.column_stack([x1, y1])

# --- ex1data2: multivariate (population, avg_income -> profit), 47 rows ---
m2 = 47
pop = rng.uniform(4.0, 22.0, size=m2)     # 10,000s
inc = rng.uniform(2.0, 9.0, size=m2)      # $10,000 (income proxy)
w2_true = np.array([0.65, 1.85])
b2_true = -4.2
noise2 = rng.normal(0, 1.5, size=m2)
y2 = pop*w2_true[0] + inc*w2_true[1] + b2_true + noise2

ex2 = np.column_stack([pop, inc, y2])

Path('data').mkdir(exist_ok=True)
np.savetxt('data/ex1data1.txt', ex1, fmt='%.4f', delimiter=',')
np.savetxt('data/ex1data2.txt', ex2, fmt='%.4f', delimiter=',')

# compute expected numbers for notebook 10 with initial w=2, b=1
w_init, b_init = 2.0, 1.0
cost = (1/(2*m1))*np.sum((w_init*x1 + b_init - y1)**2)
print('ex1data1: cost(w=2,b=1)=', float(cost))

# gradient at w=0,b=0 and at w=0.2,b=0.2 (as in notebook)
def grad_univar(x,y,w,b):
    m=len(x)
    err = (w*x+b)-y
    dj_dw = (1/m)*np.sum(err*x)
    dj_db = (1/m)*np.sum(err)
    return float(dj_dw), float(dj_db)

print('ex1data1: grad(w=0,b=0)=', grad_univar(x1,y1,0,0))
print('ex1data1: grad(w=0.2,b=0.2)=', grad_univar(x1,y1,0.2,0.2))

# run GD as notebook does (alpha=0.01, iters=1500)
w, b = 0.0, 0.0
alpha=0.01
for _ in range(1500):
    dj_dw, dj_db = grad_univar(x1,y1,w,b)
    w -= alpha*dj_dw
    b -= alpha*dj_db
print('ex1data1: gd result w,b=', float(w), float(b))