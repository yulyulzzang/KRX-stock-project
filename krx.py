#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:36:06 2023

@author: simtaeyul
"""

#모듈 설치

!pip install pmdarima
!pip install scikit-learn
!pip install finance-datareader
!pip install pandas
!pip install numpy
!pip install scikit-learn



#라이브러리 임포트

from sklearn.model_selection import train_test_split
import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pmdarima as pm
from pmdarima.arima import ndiffs
warnings.filterwarnings("ignore")




#시드 세팅

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

##############################################################################################################################################################



# 데이터 확인
train = pd.read_csv('train.csv')
train.shape

# 추론 결과를 저장하기 위한 dataframe 생성
results_df = pd.DataFrame(columns=['종목코드', 'final_return', 'rmse'])

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train['종목코드'].unique()

##############################################################################################################################################################

# 날짜 설정
start_date = datetime.strptime(str(train['일자'].min()), '%Y%m%d')
end_date = datetime.strptime(str(train['일자'].max()), '%Y%m%d')

# KOSPI 지수, USD/KRW 환율, 비트코인, oil, gold, bond 데이터 가져오기
df_kospi = fdr.DataReader('KS11', start_date, end_date)
df_USD = fdr.DataReader('USD/KRW', start_date, end_date)
df_bitcoin = fdr.DataReader('BTC/KRW', start_date, end_date)
df_oil = fdr.DataReader('OIL',  start_date, end_date)
df_gold = fdr.DataReader('GOLD', start_date, end_date)

df_bond = pd.read_csv('bond.csv')
df_bond['일자'] = pd.to_datetime(df_bond['일자'])
df_bond_sorted = df_bond[['일자', '국채_거래대금']].sort_values('일자')
df_bond_sorted.set_index('일자', inplace=True)
df_bond_sorted['국채_거래대금'] = np.log(df_bond_sorted['국채_거래대금'].str.replace(',', '').astype(float))

##############################################################################################################################################################

# 각 종목코드에 대해서 모델 학습 및 추론 반복

for code in tqdm(unique_codes):

    # 학습 데이터 생성
    train_close = train[train['종목코드'] == code][['일자', '종가', '거래량', '시가']]
    train_close['일자'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
    train_close.set_index('일자', inplace=True)
    train_close['변화량'] = train_close['종가'] - train_close['시가']

    #트레인 csv 에서는 3칼럼 사용
    tc = train_close['종가']
    volume = train_close['거래량']
    change = train_close['변화량']

    # KOSPI 지수 데이터 가져오기
    kospi_data = df_kospi[['Adj Close']].copy()
    kospi_data.rename(columns={'Adj Close': 'KOSPI'}, inplace=True)

    # USD/KRW 환율 데이터 가져오기
    usd_data = df_USD[['Adj Close']].copy()
    usd_data.rename(columns={'Adj Close': 'USD'}, inplace=True)

    # 비트코인 데이터 가져오기
    bitcoin_data = df_bitcoin[['Adj Close']].copy()
    bitcoin_data.rename(columns={'Adj Close': 'Bitcoin'}, inplace=True)

    # OIL 데이터 가져오기
    oil_data = df_oil[['Adj Close']].copy()
    oil_data.rename(columns={'Adj Close': 'Oil'}, inplace=True)

    # GOLD 데이터 가져오기
    gold_data = df_gold[['Adj Close']].copy()
    gold_data.rename(columns={'Adj Close': 'Gold'}, inplace=True)

    # 국채 거래대금 가져오기
    bond_data = df_bond_sorted[['국채_거래대금']].copy()
    bond_data.rename(columns={'국채_거래대금': 'Bond'}, inplace=True)

    # 종목 데이터와 나머지 데이터 병합하기
    merged_data = pd.merge(train_close, kospi_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, usd_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, bitcoin_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, oil_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, gold_data, left_index=True, right_index=True, how='outer')
    merged_data = pd.merge(merged_data, bond_data, left_index=True, right_index=True, how='outer')

    # 트레인 데이터의 행을 인덱스로 설정하고 나머지 잘라주기
    merged_data = merged_data.loc[train_close.index]

    # 선형 보간
    merged_data['KOSPI'] = merged_data['KOSPI'].interpolate(method='linear')
    merged_data['USD'] = merged_data['USD'].interpolate(method='linear')
    merged_data['Bitcoin'] = merged_data['Bitcoin'].interpolate(method='linear')
    merged_data['Oil'] = merged_data['Oil'].interpolate(method='linear')
    merged_data['Gold'] = merged_data['Gold'].interpolate(method='linear')
    merged_data['Bond'] = merged_data['Bond'].interpolate(method='linear')

    # 학습 데이터 다시 분리
    tc = merged_data['종가']
    volume = merged_data['거래량']
    change = merged_data['변화량']
    kospi = merged_data['KOSPI']
    usd = merged_data['USD']
    bitcoin = merged_data['Bitcoin']
    oil = merged_data['Oil']
    gold = merged_data['Gold']
    bond = merged_data['Bond']

    valid_indices = ~pd.concat([tc, volume, change, kospi, usd, bitcoin, oil, gold, bond], axis=1).isnull().any(axis=1)
    filtered_tc = tc[valid_indices]
    filtered_volume = volume[valid_indices]
    filtered_change = change[valid_indices]
    filtered_kospi = kospi[valid_indices]
    filtered_usd = usd[valid_indices]
    filtered_bitcoin = bitcoin[valid_indices]
    filtered_oil = oil[valid_indices]
    filtered_gold = gold[valid_indices]
    filtered_bond = bond[valid_indices]

    # 모델 선언, 학습시키기
    model = ARIMA(filtered_tc, exog=pd.concat([filtered_volume, filtered_change, filtered_kospi, filtered_usd, filtered_bitcoin, filtered_oil, filtered_gold, filtered_bond], axis=1), order=(1, 1, 1))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=15, exog=pd.concat([filtered_volume[-15:], filtered_change[-15:], filtered_kospi[-15:], filtered_usd[-15:], filtered_bitcoin[-15:], filtered_oil[-15:], filtered_gold[-15:], filtered_bond[-15:]], axis=1))

    # 예측값 반환
    final_return = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0]*100

    # 결과 저장
    results_df = results_df.append({'종목코드': code, 'final_return': final_return}, ignore_index=True)

results_df['순위'] = results_df['final_return'].rank(method='first', ascending=False).astype('int')

results_df

##############################################################################################################################################################

# 차분 결정
data = merged_data['종가']
n_diffs = ndiffs(data, alpha=0.05, test='adf', max_d=3)
print("차분차수 d =",n_diffs)

# auto 아리마
model = pm.auto_arima(y = data, d=1, start_p=0, max_p=3, start_q=0, max_q=3, m=1, seasonal=False, stepwise=True, trace=True)
print(model_fit.summary())

# 잔차 검정
model_fit.plot_diagnostics()
plt.show()

##############################################################################################################################################################

# 상관행렬 분석
correlation_matrix = merged_data[['종가', '거래량', '변화량', 'KOSPI', 'USD', 'Bitcoin', 'Oil', 'Gold', 'Bond']].corr()
print(correlation_matrix)

# VIF 계산
vif = pd.DataFrame()
vif["Variable"] = correlation_matrix.index
vif["VIF"] = [variance_inflation_factor(correlation_matrix.values, i) for i in range(len(correlation_matrix))]
print(vif)

##############################################################################################################################################################

# 결과 저장하기
results_df.to_csv(file_path + 'results_df.csv', index=False)
results_submission =results_df[['종목코드']].merge(results_df[['종목코드', '순위']], on='종목코드', how='left')
results_submission.to_csv(file_path + 'results_submission.csv', index=False)