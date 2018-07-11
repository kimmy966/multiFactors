import numpy as np
import pandas as pd

import numpy.linalg as nlg
import scipy.stats as stats
from sklearn import linear_model
import statsmodels.api as sm


class GRS(object):

    def __init__(self, startDate, endDate, frequency='weekly', universe_y=set_universe('000906.ZICN')):
        ###----传入参数----
        self.frequency = frequency  # 数据频率，只要日度，周度，月度三个选择
        self.universe_y = universe_y  # 股票收益溢价的股票池

        ###----自定义参数----
        self.K = 1  # 因子个数
        self.M = 0  # rf的处理
        self.datelist = self.get_datelist(startDate, endDate)  # 传入开始和结束日期，得到交易日列表list
        self.stock_premium = self.get_stock_premium()  # 得到股票收益溢价的数据dict
        self.delete_list = []  # 交易日期长度不够时需要剔除该股票

    ###得到日期列表
    def get_datelist(self, begindate, enddate):
        from CAL.PyCAL import *
        data = DataAPI.TradeCalGet(exchangeCD=u"XSHG", beginDate=begindate, endDate=enddate,
                                   field=['calendarDate', 'isWeekEnd', 'isMonthEnd'])
        if self.frequency == 'daily':
            self.M = 250
            date = map(lambda x: x[0:4] + x[5:7] + x[8:10], data['calendarDate'].values.tolist())
        if self.frequency == 'weekly':
            self.M = 52
            data = data[data['isWeekEnd'] == 1]
            date = map(lambda x: x[0:4] + x[5:7] + x[8:10], data['calendarDate'].values.tolist())
        if self.frequency == 'monthly':
            self.M = 12
            data = data[data['isMonthEnd'] == 1]
            date = map(lambda x: x[0:4] + x[5:7] + x[8:10], data['calendarDate'].values.tolist())
        return date

    ###得到股票收益的风险溢价
    def get_stock_premium(self):
        datelist = self.datelist
        stock_premium = {}  # 存为字典，键为股票secID
        for stk in self.universe_y:
            tmp = pd.DataFrame()
            for i in range(len(datelist) - 1):
                try:
                    data1 = DataAPI.MktEqudAdjGet(tradeDate=datelist[i], secID=stk, field='closePrice')
                    data2 = DataAPI.MktEqudAdjGet(tradeDate=datelist[i + 1], secID=stk, field='tradeDate,closePrice')
                    data2['tradeDate'] = map(lambda x: x[0:4] + x[5:7] + x[8:10], data2['tradeDate'].values.tolist())
                    rf = DataAPI.ChinaDataInterestRateInterbankRepoGet(indicID=u"M120000068", beginDate=datelist[i + 1],
                                                                       endDate=datelist[i + 1], field=u"dataValue")
                    data2.ix[0, 1] = data2.ix[0, 1] * 1.0 / data1.ix[0, 0] - 1 - rf.ix[0, 0] / 100 / self.M
                    data2.set_index('tradeDate', inplace=True)
                    data2.columns = ['return']
                    tmp = pd.concat([tmp, data2], axis=0)
                except:
                    continue
            stock_premium[stk] = tmp
        return stock_premium

    ###计算市场因子风险溢价数据，返回的是DataFrame，index是日期
    def get_mktfactor(self, index_name='000906.ZICN'):  # 传入作为市场因子的参考股指
        date = self.datelist
        mkt_return = pd.DataFrame(index=date[1:], columns=['mkt_premium'], data=0)
        for i in range(len(date) - 1):
            index1 = DataAPI.MktIdxdGet(tradeDate=date[i], indexID=index_name, field=u"closeIndex")
            index2 = DataAPI.MktIdxdGet(tradeDate=date[i + 1], indexID=index_name, field=u"closeIndex")
            rf = DataAPI.ChinaDataInterestRateInterbankRepoGet(indicID=u"M120000068", beginDate=date[i + 1],
                                                               endDate=date[i + 1], field=u"dataValue")  # 当月无风险收益
            mkt_return.ix[date[i + 1], 0] = index2['closeIndex'][0] * 1.0 / index1['closeIndex'][0] - 1 - \
                                            rf['dataValue'][0] * 1.0 / 100 / self.M
        return mkt_return

    '''计算DataAPI里面存在的因子风险溢价数据，返回的是DataFrame，index是日期
    factor_name:选取从DataAPI里调取哪个因子，默认为'PB'
    above_0:选择是否筛选因子数据大于0,默认为是
    inverse：选择是否对因子数据取倒数，比如PB取倒数就是账面市值比，默认是否
    up_quantile：分组时取因子数据前up%，默认前20%
    down_quantile：分组时取因子数据后down%,默认后20%
    up_minus_down：选择是否用前up%的股票组合市值加权收益减后down%的股票组合市值加权收益作为风险溢价，默认是
    universe：计算因子风险溢价的股票池，默认沪深300成分股
    '''

    def get_factor_premium(self, factor_name='PB', above_0=True, inverse=False, up_quantile=0.2, down_quantile=0.2,
                           up_minus_down=True, universe=set_universe('HS300')):
        date = self.datelist
        factor_premium = pd.DataFrame(index=date[1:], columns=[factor_name], data=0)
        for i in range(len(date) - 1):
            factorfield = list(['secID']) + list([factor_name])
            data = DataAPI.MktStockFactorsOneDayGet(tradeDate=date[i], secID=universe, field=factorfield).dropna()
            data.set_index('secID', inplace=True)
            if above_0:
                data = data[data > 0].dropna()
            if inverse:
                data = 1.0 / data
                data = data.dropna()
            up = data.quantile(up_quantile)[0]
            down = data.quantile(1 - down_quantile)[0]
            group1 = list(data[data[factor_name] < up].index)
            group2 = list(data[data[factor_name] > down].index)
            return_group1 = self.get_return(group1, date[i], date[i + 1])
            return_group2 = self.get_return(group2, date[i], date[i + 1])
            if up_minus_down:
                factor_premium.ix[i, 0] = return_group1 - return_group2
            else:
                factor_premium.ix[i, 0] = return_group2 - return_group1
        return factor_premium

    # 计算市值因子溢价，返回的是DataFrame，index是日期
    def get_mktsize_premium(self, up_quantile=0.2, down_quantile=0.2, universe=set_universe('HS300')):
        date = self.datelist
        mktsize_premium = pd.DataFrame(index=date[1:], columns=['mktvalue'], data=0)
        for i in range(len(date) - 1):
            data = DataAPI.MktEqudAdjGet(tradeDate=date[i], secID=universe, field='secID,marketValue').dropna()
            data.set_index('secID', inplace=True)
            up = data.quantile(up_quantile)[0]
            down = data.quantile(1 - down_quantile)[0]
            group1 = list(data[data['marketValue'] < up].index)
            group2 = list(data[data['marketValue'] > down].index)
            return_group1 = self.get_return(group1, date[i], date[i + 1])
            return_group2 = self.get_return(group2, date[i], date[i + 1])
            mktsize_premium.ix[i, 0] = return_group1 - return_group2
        return mktsize_premium

    # 得到投资组合的市值加权收益率
    def get_return(self, group, basedate, enddate):
        base_inf = DataAPI.MktEqudAdjGet(tradeDate=basedate, secID=group, field=u"secID,closePrice").set_index('secID')
        end_inf = DataAPI.MktEqudAdjGet(tradeDate=enddate, secID=group,
                                        field=u"secID,marketValue,closePrice").set_index('secID')
        Return = pd.concat([end_inf, base_inf], axis=1)
        Return.columns = ['Weight', 'Return', 'WReturn']  # 计算每只股票收益率和市值加权的权重以及两者的乘积
        Return['Weight'] = Return['Weight'] * 1.0 / Return['Weight'].sum()
        Return['Return'] = Return['Return'] * 1.0 / Return['WReturn'] - 1
        Return['WReturn'] = Return['Weight'] * Return['Return']
        return Return['WReturn'].sum()

    # 对股票或股票组合作回归
    def linear_regression(self, factor_data):  # 传入因子风险溢价数据
        self.K = factor_data.shape[1]
        fbar = np.array(factor_data.mean())
        omegahat = factor_data.cov().values
        bo_hat = pd.DataFrame(index=self.universe_y, columns=['intercept'], data=0)
        e = {}
        # 有多少个股票，做多少个回归
        for stk in self.universe_y:
            data = pd.concat([self.stock_premium[stk], factor_data], axis=1)
            data = data.dropna()

            x = np.zeros((self.K, len(data)))
            for i in range(self.K):
                x[i] = np.array(np.mat(data.iloc[:, [i + 1]]).reshape(len(data), ))[0]

            x = x.T
            x = sm.add_constant(x, has_constant='skip')
            y = np.array(data['return'])
            model = sm.OLS(y, x)
            results = model.fit()
            bo_hat.ix[stk, 0] = results.params[0]
            if len(data) < len(factor_data):
                self.delete_list.append(stk)
                continue
            e[stk] = results.resid

        return fbar, omegahat, bo_hat, e

    '''T = total number of observations
    N = Number of portfolios or assets
    K = Number of factors in the flist 
    fbar= column vector of the factor means (K*1)
    omegahat = variance-covariance matrix of the factors (K*K)
    bohat = column vector of intercept estimates (N *1)
    sigmahat = the residual variance-covariance matrix (N *N)'''

    # 作GRS检验
    def GRS_test(self, fbar, omegahat, bo_hat, e, alpha=0.05):
        T = len(self.datelist) - 1
        N = len(self.universe_y) - len(self.delete_list)
        K = self.K
        if T - N - K < 0:
            print
            '由于T-N-K<0,不能进行GRS检验'
            return
        fbar = np.mat(fbar).reshape(K, 1)
        omegahat = np.mat(omegahat).reshape(K, K)
        e = pd.DataFrame(e, index=np.arange(T))
        bohat = pd.DataFrame(index=np.arange(N), columns=['intercept'], data=0)

        for i in range(N):
            bohat.ix[i, 0] = bo_hat.ix[e.columns[i], 0]

        bohat = np.mat(bohat.values).reshape(N, 1)
        sigmahat = np.mat(e.cov().values).reshape(N, N)

        omega_mat = nlg.inv(omegahat)
        sigma_mat = nlg.inv(sigmahat)
        GRS = (T - N - K) * 1.0 / N * (bohat.T * sigma_mat * bohat) / (1 + fbar.T * omega_mat * fbar)[0][0]
        F = stats.f.isf(alpha, N, T - N - K)  # 自由度为N,T-N-K，显著水平alpha%下的F分位值
        return GRS, F

test = GRS('20140901','20160901',universe_y=set_universe('SH50'))  #研究时间范围：20140901~20160901，使用周度数据
factor_1 = test.get_mktfactor(index_name='000300.ZICN')         #市场因子用沪深300指数，计算其风险溢价
factor_2 = test.get_factor_premium(factor_name='PB',inverse=True)  #取1/PB为账面市值比因子，计算其风险溢价时使用默认的沪深300股票池
factor_3 = test.get_mktsize_premium(universe=set_universe('HS300')) #市值因子，计算其风险溢价时使用默认的沪深300股票池
factor_data = pd.concat([factor_1,factor_2,factor_3],axis=1).fillna(0) #把因子风险溢价拼在一起
fbar,omegahat,bo_hat,e = test.linear_regression(factor_data)    #作回归
GRS,F = test.GRS_test(fbar,omegahat,bo_hat,e)            #作GRS检验