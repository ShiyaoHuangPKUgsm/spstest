# spstest: A smooth specification test for the propensity score

## Introduction
The propensity score is perhaps one of the most fundamental concepts in modern causal inference and plays a pivotal role in identifying various causal parameters of interest.
Despite its popularity, a primary concern about the propensity score is that its form is unknown in real-world applications and must be estimated.
As pointed out by numerous studies, propensity score misspecification may lead to misleading results and erroneous conclusions about the ATE. 
Therefore, specification tests for the propensity score are vital to ensure the validity of treatment effect estimates. 

Inspired by Neyman's smooth methods, we propose to assess the propensity score specification by testing the joint significance of the generalized Fourier coefficients.
Under the null hypothesis, the test statistic is asymptotically $\chi^2$ distributed. 
Compared with existing tests, our proposed methods are computationally efficient and more sensitive to high-frequency alternatives.
Here, we provide the Python package that enables empirical researchers to use the methods conveniently.

## Installing and using spstest
To install the package, first run `pip install git+https://github.com/ShiyaoHuangPKUgsm/spstest.git` in the terminal, 
and then import it via `from spstest import spstest` in Python. Suppose we observe $n$ i.i.d. copies $\\{(D_i, X_i)\\}_{i=1}^n$ from the underlying population, where $D_i \in \\{0,1\\}$ is a treatment indicator for individual $i$, and $X_i \in \mathbb{R}^k$ is a $k$-dimensional vector of covariates.

The function spstest has six parameters:
```python
spstest(D, X, model="probit", basis="T", data_driven=True, s_range=[1, 8])
```
Specifically, `D` is an $n$-dimensional NumPy array $D=(D_1,D_2,\ldots,D_n)'$, and `X` is an $n \times k$ NumPy matrix $X = (X_1',X_2',\ldots,X_n')'$. **Note that `X` should exclude the intercept term, as it is automatically included by the function.**
The argument `model` takes either `'probit'` or `'logit'`, indicating the parametric specification to be tested. The argument `basis` specifies the set of basis functions used to construct the test statistic; we provide `'T'` for trigonometric bases and `'L'` for Legendre polynomials.
Users can set `data_driven = True` to perform the data-driven test, or `data_driven = False` to conduct the fixed-$s$ test. 
In practice, the data-driven test is more recommended, as it automatically determines the testing order. If `data_driven = True`, `s_range` should be a list of two positive integers specifying the range for selecting the optimal testing order (in this case, the upper bound for `s_range` is suggested to be less than or equal to 10). Otherwise, `s_range` should be a single positive integer $s$ (we recommend $1 \leq s \leq 4$ in this case).
After running spstest, the function returns the $p$-value of the test. We reject the null hypothesis of correct specification if the $p$-value is smaller than a chosen significance level (typically 0.05).

It is worth noting that spstest should be regarded as a model validation procedure rather than a model selection method. For a given propensity score model, spstest is designed to assess its reliability. If the null hypothesis is rejected, one may reconsider the parametric specification or modify the set of covariates by adding or removing certain variables.

## Authors
Shiyao Huang, Department of Business Statistics and Econometrics, Guanghua School of Management, Peking University, Beijing, 100871, China. Email: 2401111054@gsm.pku.edu.cn.

Xiaojun Song, Department of Business Statistics and Econometrics, Guanghua School of Management, Peking University, Beijing, 100871, China. Email: sxj@gsm.pku.edu.cn.
