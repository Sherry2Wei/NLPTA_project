import feather
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import os
import seaborn as sns
os.chdir(r"D:\lecture\QT\assignment1")
#%% input data
def end_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)

data_source = feather.read_dataframe('alans_momentum_factors.feather')
data_source.date = pd.to_datetime(data_source.date)
#%% task 1
## task1.1
print("The corrlation between ID and ret_t2_t12 : \n" + str(data_source[['ID', 'ret_t2_t12']].corr()))

# task 1.2
print("The mean and std of ID and ret_t2_t12 are the following : ")
print(data_source[['ID', 'ret_t2_t12']].agg(["mean", "std"]))
ID_percentile_list = [np.percentile(data_source["ID"].dropna(), i) for i in [1, 5, 50, 95, 99]]
ret_t2_t12_percentile_list = [np.percentile(data_source["ret_t2_t12"].dropna(), i) for i in [1, 5, 50, 95, 99]]
print("The 1st, 5th, 50th, 95th and 99th percentile for ID :\n" + str(ID_percentile_list))
print("The 1st, 5th, 50th, 95th and 99th percentile for ret_t2_t12 :\n" + str(ret_t2_t12_percentile_list))
#%% task2 1-4
## task 2.1
data_filter = data_source.query('(abs(prc_lag1) >= 5) & '
                                '(prc_lag13 == prc_lag13) & '
                                '(mcap_lag1 == mcap_lag1) & '
                                '(ret_lag2 == ret_lag2) & '
                                '(rollvalidobs >= 8)')
data_filter = data_filter.dropna(subset=['rollret_lag1', 'ID_lag2', 'ret_use'])
#%% task 2.2
def apply_quantiles(x, bins=5):
    interval = 100 / bins
    x_percentile = [np.percentile(x.dropna(), i) for i in np.arange(0, 101, interval)]
    x_percentile[0] -= 1
    x_percentile[-1] += 1
    inds = np.digitize(x, x_percentile)
    inds = pd.Series(inds, index=x.index)
    return inds

data_filter['ret_bins'] = data_filter.groupby('end_of_month')['rollret_lag1'].apply(apply_quantiles)
data_filter['ID_bins'] = data_filter.groupby(['end_of_month', 'ret_bins'])['ID_lag2'].apply(apply_quantiles)
data_filter['shrout_bins'] = data_filter.groupby('end_of_month')['shrout'].apply(apply_quantiles)
data_filter.reset_index(drop=True)
# data processing for event time return analysis

data_filter['lead0'] = data_filter['ret_use']
for i in range(1, 26):
    data_filter['lead' + str(i)] = data_filter.groupby('permno')['lead0'].shift(-i)
#%%task 2.3 creat portfolio

# HID strategy dateframe (strategy 1):
P_mom_HID = data_filter[(data_filter['ret_bins'] == 5) & (data_filter['ID_bins'] == 5)]
S_mom_HID = data_filter[(data_filter['ret_bins'] == 1) & (data_filter['ID_bins'] == 5)]

# LID strategy dataframe (strategy 2:
L_mom_LID = data_filter[(data_filter['ret_bins'] == 5) & (data_filter['ID_bins'] == 1)]
S_mom_LID = data_filter[(data_filter['ret_bins'] == 1) & (data_filter['ID_bins'] == 1)]

# regular momentum strategy
L_mom = data_filter.query('ret_bins == 5')
S_mom = data_filter.query('ret_bins == 1')

# task 2.4

def weighted_average(df, x, weight):
    return (df[x] * df[weight]).sum() / df[weight].sum()

def ret_merge(L_mom_ID, S_mom_ID):
    P_L_ID = L_mom_ID[['ret_use', 'mcap_lag1', 'end_of_month']]
    P_S_ID = S_mom_ID[['ret_use', 'mcap_lag1', 'end_of_month']]
    P_L_ID = pd.DataFrame(P_L_ID.groupby('end_of_month').apply(weighted_average, 'ret_use', 'mcap_lag1'),
                          columns=['monthly_ret'])
    P_S_ID = pd.DataFrame(P_S_ID.groupby('end_of_month').apply(weighted_average, 'ret_use', 'mcap_lag1'),
                          columns=['monthly_ret'])
    ret_merge = P_L_ID.merge(P_S_ID, left_index=True, right_index=True, suffixes=['_long', '_short'])
    ret_merge['return'] = ret_merge['monthly_ret_long'] - ret_merge['monthly_ret_short']
    ret_merge['return'][ret_merge['return'] <= -1] = 0
    ret_merge['cumulative_return'] = (ret_merge['return'] + 1).cumprod()
    return ret_merge

P_HID_r = ret_merge(P_mom_HID, S_mom_HID)
P_LID_r = ret_merge(L_mom_LID, S_mom_LID)
P_mom_r = ret_merge(L_mom, S_mom)


#%% task 4
#ff = feather.read_dataframe('ff_four_factor.feather')
#ff.columns = ['dt', 'mkt_rf', 'smb', 'hml', 'rf', 'mom']
#def plot_and_benchmark(P_HID_r, P_LID_r,P_mom_r):
    # 4.a-c
print('H_ID_mean', np.mean(P_HID_r['return']), 'H_ID_volatility', np.std(P_HID_r['return']))
sharpe_ratio = np.mean(P_HID_r['return']) / np.std(P_HID_r['return'])
print('H_ID_sharpe ratio annual', sharpe_ratio * math.sqrt(12))

print('L_ID_mean', np.mean(P_LID_r['return']), 'L_ID_volatility', np.std(P_LID_r['return']))
sharpe_ratio = np.mean(P_LID_r['return']) / np.std(P_LID_r['return'])
print('L_ID_sharpe ratio annual', sharpe_ratio * math.sqrt(12))

print('mom_mean',np.mean(P_mom_r['return']),'mom_volatility',np.std(P_mom_r['return']))
sharpe_ratio = np.mean(P_mom_r['return']) / np.std(P_mom_r['return'])
print('mom_sharpe ratio annual', sharpe_ratio * math.sqrt(12))

# 4.e plot

x = pd.to_datetime(P_HID_r.index)
y = np.log(P_HID_r['cumulative_return']+1)
x2 = pd.to_datetime(P_LID_r.index)
y2 = np.log(P_LID_r['cumulative_return']+1)
x3 = pd.to_datetime(P_mom_r.index)
y3 = np.log(P_mom_r['cumulative_return']+1)
plt.figure()
plt.plot(x, y, label='HID momentum', color='coral',linestyle=':')
plt.plot(x2, y2, label='LID momentum', color='blue')
plt.plot(x3,y3,label = 'regular momentum',color = 'red',linestyle = '-.')
plt.ylabel("log(cumulative return+1)")
plt.xlabel("date")
plt.legend()
plt.show()

# 4.d Benchmark

ff = feather.read_dataframe('ff_four_factor.feather')  # insert it to the plot function
ff.columns = ['dt', 'mkt_rf', 'smb', 'hml', 'rf', 'mom']
ff = ff[1:]
ff[ff.columns[1:]] = ff[ff.columns[1:]].apply(lambda x: 1 + x)

ff.dt = pd.to_datetime(ff.dt)
monthly_ff = ff.copy()
keys = list(monthly_ff.keys())
keys.remove('dt')
monthly_ff[keys] = 1 + monthly_ff[keys]
monthly_ff = monthly_ff.groupby(pd.Grouper(key='dt',freq='M')).prod()
monthly_ff[keys] = monthly_ff[keys] -1

HID_benchmark = pd.merge(P_HID_r, monthly_ff, left_index=True, right_index=True)
print('HID_Covariance with the market', np.cov(HID_benchmark['return'], HID_benchmark['mkt_rf'])[0][1])

LID_benchmark = pd.merge(P_LID_r, monthly_ff, left_index=True, right_index=True)
print('LID_Covariance with the market', np.cov(LID_benchmark['return'], LID_benchmark['mkt_rf'])[0][1])

y1 = HID_benchmark['return']
y2 = LID_benchmark['return']
x = HID_benchmark[['mkt_rf', 'hml', 'smb']]
model1 = sm.OLS(y1, sm.add_constant(x)).fit()
model2 = sm.OLS(y2, sm.add_constant(x)).fit()
print("HID_statistic_with_benchmark", model1.summary())
print("LID_statistic_with_benchmark", model2.summary())

#plot_and_benchmark(P_HID_r, P_LID_r,P_mom_r)

#%% task 2.5

def event_time_return(event_long,event_short):

    event_long_1 = pd.DataFrame(event_long.groupby('end_of_month').apply(weighted_average, 'ret_use', 'mcap_lag1'), columns=['lead0'])
    for i in range(1, 26):
        event_long_1['lead'+str(i)] = event_long.groupby('end_of_month').apply(weighted_average, 'lead'+str(i), 'mcap_lag1')

    event_short_1 = pd.DataFrame(event_short.groupby('end_of_month').apply(weighted_average, 'ret_use', 'mcap_lag1'), columns=['lead0'])
    for i in range(1, 26):
        event_short_1['lead'+str(i)] = event_short.groupby('end_of_month').apply(weighted_average, 'lead'+str(i), 'mcap_lag1')

    event = event_long_1 - event_short_1
    event = pd.DataFrame(event.mean(), columns=['return'])

    event['cumulative_return'] = np.cumprod(1+event['return'])

    return event
#%% task 2.5 plot the graph
HID_event = event_time_return(P_mom_HID, S_mom_HID)
LID_event = event_time_return(L_mom_LID, S_mom_LID)

plt.figure()
plt.title('return')
x = range(0,26)
y1 = HID_event['return']
y2 = LID_event['return']
plt.plot(x,y1,color = 'blue',linestyle = ':',label = 'HID')
plt.plot(x,y2,color = 'red',label = 'LID')
plt.legend()
plt.show()

plt.figure()
plt.title('cumulative return')
x = range(0,26)
y1 = HID_event['cumulative_return']-1
y2 = LID_event['cumulative_return']-1
plt.plot(x, y1,color = 'coral',linestyle = '-.',label = 'HID')
plt.plot(x,y2,color = 'blue',label = 'LID')
plt.legend()
plt.show()


#%% task 2.6
def liquidity(L_condition, S_condition):
    L_condition_liquidity = L_condition.groupby('end_of_month').apply(weighted_average, 'shrout_bins', 'mcap_lag1')
    S_condition_liquidity = S_condition.groupby('end_of_month').apply(weighted_average, 'shrout_bins', 'mcap_lag1')
    L_condition_liquidity = L_condition_liquidity.reset_index(name="weight-average-shrout-bin")
    S_condition_liquidity = S_condition_liquidity.reset_index(name="weight-average-shrout-bin")
    return L_condition_liquidity, S_condition_liquidity


def liquidity_plot(L_condition, S_condition):
    sns.distplot(L_condition["weight-average-shrout-bin"], color="skyblue", label="Long High momentum")
    sns.distplot(S_condition["weight-average-shrout-bin"], color="red", label="Short Low momentum")
    plt.legend()
    plt.show()


# strategy 1
P_mom_HID_liquid, S_mom_HID_liquid = liquidity(P_mom_HID, S_mom_HID)
liquidity_plot(P_mom_HID_liquid, S_mom_HID_liquid)
# strategy 2
L_mom_LID_liquid, S_mom_LID_liquid = liquidity(L_mom_LID, S_mom_LID)
liquidity_plot(L_mom_LID_liquid, S_mom_LID_liquid)


#%% task 3.1
y_HID = P_HID_r['return']
x_mom = P_mom_r['return']
model = sm.OLS(y_HID, sm.add_constant(x_mom)).fit()
print(model.summary())

#%% task 3.3
def analyze(strategy_dataframe):
    ff_dataframe = feather.read_dataframe('ff_four_factor.feather')
    ff_dataframe.columns = ff_dataframe.columns.str.lower()
    ff_dataframe = ff_dataframe.loc[1:, ]  # remove first empty string
    ff_dataframe['date'] = ff_dataframe.dt.apply(lambda x: end_of_month(datetime.datetime.strptime(x, '%Y-%m-%d')))
    ff_dataframe = ff_dataframe.groupby(['date'])[['mkt_rf', 'hml', 'smb', 'mom']].apply(lambda x : (1 + x).prod() - 1)
    ff_dataframe["mom_factor_cumulative_return"] = ((1 + ff_dataframe.mom).cumprod()).apply(np.log10)

    strategy_benchmark = pd.merge(strategy_dataframe, ff_dataframe, left_index=True, right_index=True)
    y = strategy_benchmark['return']
    x = strategy_benchmark[["mkt_rf", "smb", "hml", "mom"]]
    model = sm.OLS(y, sm.add_constant(x)).fit()
    print("Continuous_momentum_factor_statistic_with_benchmark", model.summary())

    print('continuous momentum factor_mean', np.mean(strategy_dataframe['return']),
          'continuous momentum factor_volatility', np.std(strategy_dataframe['return']))
    sharpe_ratio = np.mean(strategy_dataframe['return']) / np.std(strategy_dataframe['return'])
    print('continuous momentum factor_sharpe ratio annual', sharpe_ratio * math.sqrt(12))

    print('mom_factor_mean', np.mean(ff_dataframe['mom']), 'mom_factor_volatility', np.std(ff_dataframe['mom']))
    sharpe_ratio = np.mean(ff_dataframe['mom']) / np.std(ff_dataframe['mom'])
    print('mom_factor_sharpe ratio annual', sharpe_ratio * math.sqrt(12))

    x = pd.to_datetime(strategy_dataframe.index)
    y = np.log(P_HID_r['cumulative_return'] + 1)
    x2 = pd.to_datetime(ff_dataframe.index)
    y2 = np.log(ff_dataframe["mom_factor_cumulative_return"] + 1)
    plt.figure()
    plt.plot(x, y, label='continuous momentum factor', color='coral', linestyle=':')
    plt.plot(x2, y2, label=' Fama French momentum factor', color='blue')
    plt.ylabel("log(cumulative return+1)")
    plt.xlabel("date")
    plt.legend()
    plt.show()

analyze(P_LID_r)

#%% Task1
    # task 1.1
def mcap_plot(strategy1, strategy2,regular_mon):
    sns.distplot(strategy1["mcap_lag1"],
                 color="skyblue", label="Strategy-1", hist=False)
    sns.distplot(strategy2["mcap_lag1"],
                 color="red", label="strategy-2", hist=False)
    sns.distplot(regular_mon["mcap_lag1"],
                 color="gold", label="regular-mom", hist=False)
    strategy1_min = min(strategy1["mcap_lag1"])
    strategy2_min = min(strategy2["mcap_lag1"])
    regular_mom_min = min(regular_mon["mcap_lag1"])
    x_min = min([strategy1_min,regular_mom_min])
    strategy1_max = max(strategy1["mcap_lag1"])
    strategy2_max = max(strategy2["mcap_lag1"])
    regular_mom_max = max(regular_mon["mcap_lag1"])
    x_max = max([strategy1_max, regular_mom_max])
    plt.xlim(x_min, x_max)
    plt.legend()
    plt.show()

P_HID = pd.concat([P_mom_HID,S_mom_HID],axis = 0)
P_LID = pd.concat([L_mom_LID,S_mom_LID],axis = 0)
P_mom = pd.concat([L_mom,S_mom],axis = 0)

mcap_plot(P_HID,P_LID,P_mom)















