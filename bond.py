import numpy as np
import scipy.optimize as sciopt

p = 9700 # 채권매입가격

var = dict(
    n_payment_year = 4, # 연 이자지급횟수
    coupon = 400 / n_payment_year, # 이표율 4% 가정 /  연 4회 지급 기준
    principal = 10000, # 원금
    n_intervals = 7,   # 향후 남은 이자지급횟수
    day = 44, # 다음 이자지급일까지 남은 일할계산
    day_interval = 91  # 금번 이자지급기간의 총 일수
)

#1. 채권가격

def bond_price(ytm, var):

    ytm = ytm/var['n_payment_year']
    
    cf_total = [var['coupon']] * var['n_intervals']
    pv_total = []

    first_factor = 1 / (1 + ytm * var['day'] / var['day_interval'])
    first_cf = cf_total.pop() * first_factor
    pv_total.append(first_cf)

    for i, cf in enumerate(cf_total):
        next_cf =  cf * first_factor * 1 / ((1 + ytm) ** (i + 1))
        pv_total.append(next_cf)

    pv_principal = var['principal'] * first_factor * 1 / ((1 + ytm) ** (var['n_intervals'] - 1))

    return pv_principal + sum(pv_total)

#2. 매수가격에서 YTM 역산

def dif_price(ytm, var, p):
    res = np.abs(bond_price(ytm, var) - p)
    return res

res = sciopt.minimize(dif_price, x0 = 0.05, args = (var, p), bounds = [(0, np.inf)])



