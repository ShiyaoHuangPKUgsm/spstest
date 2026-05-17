# spstest: A smooth specification test for the propensity score

## Introduction
The propensity score is perhaps one of the most fundamental concepts in modern causal inference and plays a pivotal role in identifying various causal parameters of interest.
Despite its popularity, a primary concern about the propensity score is that its form is unknown in real-world applications and must be estimated.
As pointed out by numerous studies, propensity score misspecification may lead to misleading results and erroneous conclusions about the ATE. 
Therefore, specification tests for the propensity score are vital to ensure the validity of treatment effect estimates. 

Inspired by Neyman's smooth methods, we propose to assess the propensity score specification by testing the joint significance of the generalized Fourier coefficients.
Under the null hypothesis, the test statistic is asymptotically $\chi^2$ distributed. 
Compared with existing tests, our proposed methods are computaionally efficient and more sensitive to high-frequency alternatives.
Here, we provide th python package that enables empirical researchers to use the methods conveniently.

## Installing and using spstest
To install the package, first run `pip install git+https://github.com/ShiyaoHuangPKUgsm/spstest.git` in the terminal, 
and then import it via `from spstest import spstest` in python. 

The function spstest has six parameters:
```python
spstest(D, X, model="probit", basis="T", data_driven=True, s_range=[1, 8])
```
