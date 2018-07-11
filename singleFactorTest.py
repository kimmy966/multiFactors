import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

class OutlierCleaner(object):

    def __init__(self,is_plot=True):
        self.is_plot=is_plot

    def __plothist(self,factor):
        plt.hist(factor.values, bins=12, color='b', alpha=.4)

    def __Nstd(self,thisGroup,N=3):
        #N倍标准差，异常值
        thisM=thisGroup.mean()
        thisS=thisGroup.std()
        cap=float(thisM+N*thisS)
        flo=float(thisM-N*thisS)
        thisGroupC=thisGroup.copy()
        thisGroupC[thisGroup>cap]=cap
        thisGroupC[thisGroup<flo]=flo
        return thisGroupC

    def Nstd_pannel(self,factor):
        #N倍标准差，面板数据
        new_factor=self.__Nstd(factor,N=3)
        if self.is_plot: self.__plothist(new_factor)
        return new_factor

class Standardization(object):

    def __init__(self):
        pass

    def __ZScore(self, thisGroup):
        thisM = float(thisGroup.mean())
        thisS = float(thisGroup.std())
        thisGroupC = thisGroup.copy()
        thisGroupC = (thisGroupC - thisM) / thisS
        print(thisGroupC)
        return thisGroupC

    '''
    在进行行业quantile标准化，出现整个行业仅有一只股票的情形时，将该股票因子赋值为1-epsilon(因其为该行业最大)
    '''
    def __QuantileChange(self,thisGroup):
        f = thisGroup.values
        if len(f) == 1:
            epsilon = 0.0001
            r = [1-epsilon]
        else:
            I = np.argsort(-f)
            g = f.copy()
            g[I] = np.arange(len(f))
            quantile = (g+1)/(len(f))
            r = ss.norm.ppf(quantile)
            r[quantile == 1] = ss.norm.ppf((len(f)-0.5)/len(f)) #将inf用一个比max(r[quantile!=1])大一些的数代替
        return r

    def QuantileChange_Ind(self, factorGroup):
#        factorGroup['industry'] = indGroup.values
#        factorGroup=factorGroup.join(indGroup,how='left')
		#分行业进行quantile标准化
        indlst = [self.QuantileChange_sectional(groupedItem[1].drop('Industry',axis = 1)) for groupedItem in factorGroup.groupby(by='Industry')]
        inddf = pd.concat(indlst, ignore_index = False)
        inddf.reset_index(inplace=True)
        inddf2=inddf.sort_values(by=['date','code'])
        inddf2.set_index(['date','code'],inplace=True)
        return inddf2

    def ZScore_Ind(self, factorGroup):
#        factorGroup['industry'] = indGroup.values
        #分行业进行zscore标准化
        indlst = [self.ZScore_sectional(groupedItem[1].drop('Industry',axis = 1)) for groupedItem in factorGroup.groupby(by='Industry')]
        inddf = pd.concat(indlst, ignore_index = False)
        inddf.reset_index(inplace=True)
        inddf2=inddf.sort_values(by=['date','code'])
        inddf2.set_index(['date','code'],inplace=True)
        return inddf2

    def ZScore_sectional(self, factor):
        new_factor = factor.groupby(by='date').transform(self.__ZScore)
        return new_factor

    def QuantileChange_sectional(self, factor):
        new_factor = factor.groupby(by='date').transform(self.__QuantileChange)
        return new_factor


class Orthogonalized(object):
    # 正交函数
    def __init__(self):
        pass

    def get(self, data, Y_columns, X_columns, Dummy_columns=[], method='OLS'):
        # data原始数据
        # Dummy_columns需要包含在X_columns里面
        if type(Y_columns) is not list: Y_columns = [Y_columns]
        if type(X_columns) is not list: X_columns = [X_columns]
        if type(Dummy_columns) is not list: Dummy_columns = [Dummy_columns]
        self.yc, self.xc, self.dc = Y_columns, X_columns, Dummy_columns

        assert ~data.isnull().any().any(), '含nan'
        #            data=data.dropna(0)

        Resid = []
        for key, group in data.groupby(by='date'):
            tempResid = self.OLS(group)
            Resid.append(tempResid)
        Resid = pd.concat(Resid)
        return Resid

    def OLS(self, data):
        y = data.loc[:, self.yc]
        x = data.loc[:, self.xc]
        if len(self.dc) > 0:
            x = pd.get_dummies(x, columns=self.dc)
        #        x=sm.add_constant(x)
        results = sm.OLS(y, x).fit()
        r = pd.DataFrame(results.resid, columns=['factor'], index=data.index)
        return r


class Regression(object):
    # 线性回归函数
    def __init__(self):
        pass

    def __summary(self, Para):
        # 回归打印总结表格
        t = Para['t'].mean()
        mu = Para['f'].mean() * 100
        pct = float((Para['t'] > 0).sum() / len(Para['t'])) * 100
        abs_t = Para['t'].abs().mean()
        x = np.array([t, mu, pct, abs_t])
        result = pd.DataFrame({'Value': np.round(x, 2), u'指标名称': [u'因子收益序列t值', u'因子收益均值 % ', u't>0比例 % ', u'abs(t)均值']})
        result = result.set_index(['指标名称'])
        print(result)

    def __draw(self, Para):
        # 回归画图函数
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 定义中文字体
        plt.figure(1)
        fig, axes = plt.subplots(1, figsize=(10, 6))
        Para['f'].hist(color='b', alpha=0.6, bins=10)
        plt.ylabel('Frequency')
        plt.legend(['f value'], loc='upper right', fontsize=15)
        plt.title(u'因子系数f值分布', fontsize=15)
        plt.grid(True, color='k', alpha=0.2)

        lenX = len(Para)
        lable = np.arange(0, lenX, 12)

        plt.figure(2)
        fig2, axes2 = plt.subplots(1, figsize=(10, 6))
        Para['f'].plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['f value'], loc='upper left', fontsize=15)
        xticklabels = Para.index[lable].strftime('%Y-%m')
        axes2.set_xticks(lable)
        axes2.set_xticklabels(xticklabels, rotation=15)
        plt.grid(True, color='k', alpha=0.2)
        plt.title(u'因子系数f值时间序列', fontsize=15)

        plt.figure(3)
        fig3, axes3 = plt.subplots(1, figsize=(10, 6))
        Para['t'].abs().plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['abs_tstat'], loc='upper right', fontsize=15)
        xticklabels = Para.index[lable].strftime('%Y-%m')
        axes3.set_xticks(lable)
        axes3.set_xticklabels(xticklabels, rotation=15)
        plt.grid(True, color='k', alpha=0.2)
        plt.title(u'回归t值得绝对值', fontsize=15)
        plt.show()

    def OLS(self, data):
        y = data.loc[:, self.retName]
        x = data.loc[:, self.factorName]
        results = sm.OLS(y, x).fit()
        f = results.params.iloc[0]
        t = results.tvalues.iloc[0]
        time = data.index[0][1]
        r = [time, f, t]
        return r

    def get(self, data, retName='futureRet', factorName='factor', method='OLS', isSummary=True, isPlot=True):
        data = data.loc[:, [retName, factorName]]
        assert ~(data.isnull().any().any()), '含nan报错'
        self.retName, self.factorName = retName, factorName
        Para = []  # 记录检验值
        for key, group in data.groupby(by='date'):
            if method == 'OLS':
                tempPara = self.OLS(group)
            else:
                raise Exception
            Para.append(tempPara)

        Para = pd.DataFrame(data=Para, columns=['time', 'f', 't']).set_index(['time'])

        if isSummary:
            self.__summary(Para)
        if isPlot:
            self.__draw(Para)

        return Para


class IC(object):
    # IC计算
    def __init__(self):
        pass

    def __summary(self, ic, t):
        # IC打印总结表格
        lenIC = len(ic)
        mu = float(ic.rankIC.mean())
        std = float(ic.rankIC.std())
        pct = float((ic.rankIC > 0).sum() / lenIC) * 100
        IR = float(mu / std)
        meanT = float(ic.t.abs().mean())
        tt = sum(ic.t.abs() > t) / lenIC
        t_pos = sum((ic.rankIC > 0) & (ic.t.abs() > t)) / sum(ic.rankIC > 0) * 100
        t_neg = sum((ic.rankIC < 0) & (ic.t.abs() > t)) / sum(ic.rankIC < 0) * 100
        x = np.array([mu * 100, std, pct, IR, meanT, tt, t_pos, t_neg])
        result = pd.DataFrame({'Value': np.round(x, 2),
                               'Name': ['IC_mean', 'IC_std', 'IC_pos_per', 'IR', 't_mean', 'sign_all', 'sign_pos',
                                        'sign_neg']})
        result = result.set_index('Name')
        return result

    def __draw(self, rankIC):
        # IC画图
        lenX = len(rankIC)
        lable = np.arange(0, lenX, 12)
        fig, axes = plt.subplots(1, figsize=(10, 6))
        rankIC['rankIC'].plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['rankIC'], loc='upper right', fontsize=15)
        x = 1
        y = max(rankIC['rankIC'])
        plt.text(x, y, r'$\mu_{IC}=$' + str(round(rankIC['rankIC'].mean() * 100, 2)) + '(%)', color='r', fontsize=15)
        xticklabels = rankIC.index[lable].strftime('%Y-%m')
        xtick = lable
        axes.set_xticks(xtick)
        axes.set_xticklabels(xticklabels, rotation=30, fontsize=15)
        plt.grid(True, color='k', alpha=0.2)
        plt.show()

    def rankIC(self, data, retName='futureRet', factorName='factor', t=1.96, isSummary=True, isPlot=True):
        data = data.loc[:, [retName, factorName]]
        assert ~(data.isnull().any().any()), '含nan报错'
        ic = data.groupby(by='date').agg(lambda x: np.array(ss.spearmanr(x)))
        ic.columns = ['rankIC', 'pValue']
        rs = ic.loc[:, 'rankIC'].values
        ic.loc[:, 't'] = rs * np.sqrt((len(data) / len(rs) - 2) / ((rs + 1.0) * (1.0 - rs)))

        # 最后一期收益率都是0，因此应该抹去
        ic = ic.iloc[:-1]
        summary = self.__summary(ic, t)
        if isSummary:
            print(summary)
        if isPlot:
            self.__draw(ic)
        return ic, summary