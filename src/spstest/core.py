import math
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.special import comb


def pai_triangle(j, x):
    if j == 0:
        return 1
    elif j % 2 == 1:
        return np.sqrt(2) * np.sin((j + 1) * np.pi * x)
    else:
        return np.sqrt(2) * np.cos(j * np.pi * x)


def pai_Legendre(j, x):
    temp = 0
    for k in range(j + 1):
        temp = temp + comb(j, k) * comb(j + k, k) * ((-x) ** k)
    return math.sqrt(2 * j + 1) * ((-1) ** j) * temp


def cal_theta(D, X, model):
    if model == "probit":
        probit_model = sm.Probit(D, X)
        probit_result = probit_model.fit(disp=False)
        return (probit_result.params).reshape(-1, 1)
    if model == "logit":
        logit_model = sm.Logit(D, X)
        logit_result = logit_model.fit(disp=False)
        return (logit_result.params).reshape(-1, 1)


def cal_Z(X, hat_theta, model):
    n = X.shape[0]
    if model == "probit":
        return np.reshape(stats.norm.cdf(X @ hat_theta, loc=0, scale=1), (n,))
    if model == "logit":
        return np.reshape(stats.logistic.cdf(X @ hat_theta, loc=0, scale=1), (n,))


def cal_g(X, hat_theta, model):
    if model == "probit":
        return (X.T) * (stats.norm.pdf(X @ hat_theta, loc=0, scale=1).T)
    if model == "logit":
        return (X.T) * (stats.logistic.pdf(X @ hat_theta, loc=0, scale=1).T)


def cal_G(j, g, Z, basis):
    if basis == "T":
        return np.mean(g * pai_triangle(j, Z), axis=1)
    if basis == "L":
        return np.mean(g * pai_Legendre(j, Z), axis=1)


def cal_Delta(g):
    n = g.shape[1]
    return (g @ g.T) / n


def cal_PI(Z, s, basis):
    n = Z.shape[0]
    PI = np.zeros((n, s))
    if basis == "T":
        pai = pai_triangle
    if basis == "L":
        pai = pai_Legendre
    for j in range(s):
        PI[:, j] = pai(j, Z)
    return PI


def cal_GM(g, Z, s, basis):
    p = g.shape[0]
    GM = np.zeros((p, s))
    for j in range(s):
        GM[:, j] = cal_G(j, g, Z, basis)
    return GM


def test_statistic(D, X, model, basis, s):
    n = X.shape[0]
    hat_theta = cal_theta(D, X, model)
    Z = cal_Z(X, hat_theta, model)
    g = cal_g(X, hat_theta, model)
    Delta = cal_Delta(g)
    GM = cal_GM(g, Z, s, basis)
    PI = cal_PI(Z, s, basis)

    BE = np.zeros((n, s))
    for j in range(s):
        BE[:, j] = (D - Z) * (PI[:, j] - np.reshape(GM[:, j:j + 1].T @ np.linalg.inv(Delta) @ g, (n,)))

    A = np.reshape(BE.mean(axis=0), (-1, 1))
    Sigma = (BE.T @ BE) / n
    return (n * A.T @ (np.linalg.inv(Sigma)) @ A)[0, 0]


def data_driven_test(D, X, model, basis, s_range):
    n = X.shape[0]
    result_ls = []
    for s in range(s_range[0], s_range[1] + 1):
        statistic = test_statistic(D, X, model, basis, s)
        result_ls.append((s, statistic, statistic - s * np.log(n)))
    result_ls.sort(key=lambda x: x[-1], reverse=True)
    return result_ls[0][1]


def H(x, n):
    if x <= np.log(n):
        return (2 * stats.norm.cdf(np.sqrt(x)) - 1) * (2 * stats.norm.cdf(np.sqrt(np.log(n))) - 1)
    elif np.log(n) < x < 2 * np.log(n):
        return H(np.log(n), n) + (x - np.log(n)) * (H(2 * np.log(n), n) - H(np.log(n), n)) / np.log(n)
    else:
        return (2 * stats.norm.cdf(np.sqrt(x)) - 1) * (2 * stats.norm.cdf(np.sqrt(np.log(n))) - 1) \
               + 2 * (1 - stats.norm.cdf(np.sqrt(np.log(n))))


def spstest(D, X, model="probit", basis="T", data_driven=True, s_range=[1, 8]):
    D = np.array(D)
    X = np.array(X)
    n = X.shape[0]
    X = np.column_stack((np.ones(n), X))

    if data_driven:
        return 1 - H(data_driven_test(D, X, model, basis, s_range), n)
    else:
        return stats.chi2.sf(test_statistic(D, X, model, basis, s_range), s_range)