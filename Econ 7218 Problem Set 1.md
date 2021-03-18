# Econ 7218 Problem Set 1 
*r09323036 經濟所碩一 李祖福*


For this course, it will be necessary to use a general or scientific programming language such as Matlab, Python, or R. The goal of this problem set is to learn the basics of such tools by simulating and estimating a simple discrete choice model via numerical optimization algorithms.

Consider a simple binary choice model where unobserved error terms U1 and U2 are both standard type-I value distributed with cdf

$$F (u) = e^{e^{(-u)}}$$

and where there are two covariates $X_1$ ∼ $N(0,1)$ and $X_2$ ∼ $χ^2(1)$.

$$f(x)=
\begin{cases}
1& if\ \ X_{1i}β_1 +U_{1i} >X_{2i}β_2 +U_{2i}\\
0& otherwise
\end{cases}$$

The resulting probability function of $y_i$ is
$$
P r(y_i = 1|X_{1i}, X_{2i}) = \dfrac{exp(X_{1i}β_1 − X_{2i}β_2) }{1 + exp(X_{1i}β_1 − X_{2i}β_2)}
$$

Suppose that $β_1$ = 1.0 and $β_2$ = −0.5.


---
### Q1. Simulate a dataset of size N = 400 from the model for a given set of parameter values ($β_1, β_2$).

$sol:$
```
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression

np.random.seed(0)

u1 = np.random.gumbel(0, 1, 400)
u2 = np.random.gumbel(0, 1, 400)
x1 = np.random.normal(0, 1, 400)
x2 = np.random.chisquare(1, 400)

beta1 = np.zeros(400) + 1
beta2 = np.zeros(400) - 0.5
```


---

### Q2. Code the log likelihood function as a function of the parameters ($β_1, β_2$).

$sol:$
```
G = np.exp(x1*beta1 - x2*beta2) / (1 + np.exp(x1*beta1 - x2*beta2))
f = (y*np.log(G) + (1 - y)*np.log(1 - G)).sum()
```


---

### Q3. Code a grid search algorithm over the parameter space $β_1 ∈ [−5, 5]$ and $β_2 ∈ [−5, 5]$.
$sol:$
```
def grid_search(n):
    beta_1 = np.array(np.arange(-5, 5, n))
    beta_2 = np.array(np.arange(-5, 5, n))

    f_max = -1000
    
    for i in beta_1:
        for j in beta_2:
            G = np.exp(x1*i - x2*j) / (1 + np.exp(x1*i - x2*j))
            f = (y*np.log(G) + (1 - y)*np.log(1 - G)).sum()
             
            if f > f_max:
                f_max = f
                
                beta1_hat = i
                beta2_hat = j


    return beta1_hat, beta2_hat
```
```
start = time.time()
beta1, beta2 = grid_search(0.025)
end = time.time()

print('Beta_1 : ' , beta1)
print('Beta_2 : ' , beta2)
print('Time : ', end - start)
```

![](https://i.imgur.com/M6kemlk.png)

---

### Q4. Generate R = 100 samples of size N = 400 in Step 1. Estimate the model for each sample with a gradient method (BHHH, BFGS, etc.) or Nelder-Mead to maximize the log likelihood function and report the mean and standard deviation of the parameter estimates across the samples.
$sol:$
```
def data_generation():
    u1 = np.random.gumbel(0, 1, 400)
    u2 = np.random.gumbel(0, 1, 400)
    x1 = np.random.normal(0, 1, 400)
    x2 = np.random.chisquare(1, 400)

    beta1 = np.zeros(400) + 1
    beta2 = np.zeros(400) - 0.5
    
    y = ((x1 * beta1) + u1) - ((x2 * beta2) + u2)
    y[y > 0] = 1
    y[y < 0] = 0
    X = np.hstack((x1.reshape(-1,1), -x2.reshape(-1,1)))
    return X, y
```
```
coef = pd.DataFrame()

for i in range(100):
    X, y = data_generation()
    clf = LogisticRegression(solver = 'lbfgs').fit(X, y)
    coe = pd.DataFrame(clf.coef_)
    coef = coef.append(coe)
```
```
beta1_mean = coef[0].mean()
beta1_std = coef[0].std()

beta2_mean = coef[1].mean()
beta2_std = coef[1].std()
```
![](https://i.imgur.com/45GIcek.png)
