#%%

import numpy as np
import compute as compute
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import scipy.stats as sstat
import arch
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

'''
변동성 추정 Process

1. 경험적 수기 조정 과정

오늘까지 실현변동성이 이 수준이고 (=평균 회귀) 
앞으로 오르/내릴 확률이 높아 (=군집성, 현재 vol score 의 추세)

평균회귀계수 : 볼이 평균보다 높거나 / 낮으면 -> 낮아지거나 높아질 가능성 고정값으로 multiplier
추세지속계수 : 볼이 현재 오르고 있거나 / 내리고 있으면 -> 계속 높아지거나 / 낮아질 가능성 고정값 multiplier

-> 2번 추정 모델들의 공통적인 문제로
변동성이 별일 없을때는 비슷하게 움직이나, 
1) spike 발생을 예측 못하는 가운데 후행해버림
2) spike 정도도 과소반영함 (garch 계열의 시계열 모형들이 특히)
3) 이어지는 변동성 축소도 평균회귀계수가 부족하여 후행
그러므로 돌려보면 Garch나 ML이나 사실상 실제 일변동성의 이동평균 느낌으로 연출됨
그러므로 억지로 모멘텀적인 요소 반영하여 오를때는 더오르고 / 내릴때는 더내리는 상황 보정

2) 실제 변동성 안 쓰고 여러 일변동성 평균낸 custom vol score 사용

static 하게 "실제 현재까지 실현변동성" 과 관련된 지표들로 구성된 vol score
해당 vol score 의 분포도에서의 현재 위치 파악 (평균으로 오를 확률 / 내릴 확률)
해당 vol score 의 plot 을 통해 현재 추세 파악 (추세 지속 가능성 파악)

-> 여기서 중요한건 현재 수준이 이정도이므로 앞으로 확대 / 축소될 가능성이 높다! 만 판단하는 것 ("실제 얼마나" 는 밑에 2번에서 추정)

----------------------------------
2. (1번에서 애초에 평균 수준으로 오를것/내릴것이라고 상정해놓고)
만약 오르/내린다면 어떤 식으로 분포를 띄는 가운데 변동성 path 형성할 것인가?

1) stochastic model 에 근거하여 추정한 변동성 -> 다음날 gaussian dist에 투입하여 수익률 도출 -> 해당 수익률을 분산으로 보고 7일간 recursive 구조 형성 -> monte carlo simulation 
2) ML model 에 근거하여 추정한 변동성 -> 다음날 gaussian dist에 투입하여 수익률 산출
-> 해당 수익률을 분산으로 보고 t+1 ML 진행 + 이러한 recursive 구조 7번 형성 -> monte carlo simulation
3) Deep learning 모델로 아예 wholesale 로 7일치 추정

----------------------------------
3. 이러한 분포로 움직인다고 했을때 내 IV가 지금 매력/감당할 수 있는 수준인가?

=> 2번에서 도출된 7일 변동성 추정치들의 분포에 내 IV 대응해서 P 구하기 -> 최종 Weight 비중으로 사용

'''

# %% vol function

class vol_forecast(compute.ohlc_return):

    ''' df = raw daily price table from outer sources in OHLC format / processed by inner myfunc.return_function
        interval = return horizon
        start_date = (optional) start_date'''

    def __init__(self, df_daily, interval, start_date = 0):
        
        super().__init__(df_daily)

        self.start_date = pd.to_datetime(start_date)
        if start_date != 0:
            self.df = df_daily.sort_index(ascending = True).loc [df_daily.index > self.start_date, :]
        else:
            self.df = df_daily.sort_index(ascending = True)
        
        self.interval = interval
        annualizing_factor = 252 / interval

        self.ret_total = self.total_return(interval)
        self.volscore_total = (self.ret_total.pipe(compute.volscore, price = 'close', n = annualizing_factor) + self.ret_total.pipe(compute.volscore, price = 'tr', n = annualizing_factor)) / 2
        self.ret_up = self.plus_return(interval)
        self.volscore_up = (self.ret_up.pipe(compute.volscore, price = 'close', n = annualizing_factor) + self.ret_up.pipe(compute.volscore, price = 'high', n = annualizing_factor)) / 2
        self.ret_down = self.minus_return(interval)
        self.volscore_down = (self.ret_down.pipe(compute.volscore, price = 'close', n = annualizing_factor) + self.ret_down.pipe(compute.volscore, price = 'low', n = annualizing_factor)) / 2

        self.dict = dict(total = self.volscore_total, up = self.volscore_up, down = self.volscore_down)
    
    def status(self, avg_days = 5):

        color = ['g', 'r', 'b']

        current_vol = dict(total = self.volscore_total[-1], up = self.volscore_up[-1], down = self.volscore_down[-1])
        current_p = dict()

        self.fig_1, axes = plt.subplots(2, 3, figsize = (20, 20))
        bins = np.linspace(0, 1, 200)

        for i, keys in enumerate(self.dict.keys()):

            df = self.dict.get(keys)
            prob = compute.custom_cdf_function(df, current_vol.get(keys))
            current_p[keys] = prob

            trend = df.tail(252)
            trend_wm = df.tail(257).rolling(5).apply(lambda x : np.dot(x, np.arange(1, avg_days + 1)) / sum(np.arange(1, avg_days + 1)))
            
            axes[0, i].hist(df, density = True, bins = bins, color = color[i])
            axes[0, i].axvline(current_vol.get(keys), color = 'black')
            axes[0, i].set_title(keys)

            axes[1, i].plot(trend, label = 'trend')
            axes[1, i].plot(trend_wm, label = 'weighted_trend')
            axes[1, i].legend()

        return current_vol, current_p, self.fig_1
    
    def forecast_garch(self, n_days = 6):
        
        fig, ax = plt.subplots(1, 1)

        if self.interval != 1:
            raise ValueError('interval is not 1 day')
        else:
            dummy = compute.mc_garch(self.ret_total['close'], n_days, start_date = self.start_date)
        
        ax.hist(dummy[0], bins = np.linspace(0, 0.5, 300))
        result = dummy[0].mean()
        ax.text(2, 2, result)

        return fig, result
    
    def forecast_ml(self):
        print("to be written")

    def final_probability(self):
        print('to be written')

def iv(df_vkospi):
    
    current = df_vkospi.iat[-1, 0]
    
    fig, ax = plt.subplots(1, 1)
    ax.hist(df_vkospi, bins = np.linspace(0, 100, 500))
    ax.axvline(current, linewidth = 2, color = 'b')
    ax.text(5, 5, str(current))
    p = compute.custom_cdf_function(df_vkospi['종가'], current)

    return fig, p

#%% 

# # 2) ML based vol prediction 향후 추가

# # 전처리
# # X : close vol, TR vol, volscore_total 3개 -> 평균 0 가정시 표준편차 = abs(수익률)
# # Y : 다음날 close vol (close_vol.shift(-1))

# predictors = ['close', 'tr', 'volscore']
# df_x = df_price[predictors]
# df_x.update(np.abs(df_x))
# df_y = np.abs df_price['close']).shift(-1)

# x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.01, shuffle = False)

# lin_reg = LinearRegression()
# lin_reg.fit(x_train, y_train)
# lin_pred = lin_reg.predict(x_test)

# xgb_reg = XGBRegressor()
# xgb_reg.fit(x_train, y_train)
# xgb_pred = xgb_reg.predict(x_test)

# rf_reg = RandomForestRegressor()
# rf_reg.fit(x_train, y_train)
# rf_pred = rf_reg.predict(x_test)


# #%%    4. 추정된 변동성 분포에서 IV 위치 파악



# # daily_vol_tr = np.sqrt df_price['TR'] ** 2) * np.sqrt(252)
# # normal_tr = func.garch_process df_price['TR'], 7, n_interval = 'day', n_paths = 1000)

# # %% 
# # weekly anaylsis

# # df_weekly = pd.read_excel("C:/Users/kanld/Desktop/rawdata_230421.xlsx", sheet_name = "weekly_data", index_col = 0)
# # df_weekly = df_weekly.sort_index(ascending = True)

# # weekly_close = func.garch_process(df_weekly['close'], 2, n_interval = 'week', n_paths = 1000)
# # weekly_tr = func.garch_process(df_weekly['TR'], 2, n_interval = 'week', n_paths = 1000)


