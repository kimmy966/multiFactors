import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn import preprocessing
import statsmodels.formula.api as sm
import math

class DataDealing(object):

    def __init__(self):
        pass

    def dealing(self,factorM,df):
        date = factorM.index
        stk = factorM.keys()

        df = pd.DataFrame(df,index=date,columns=stk)
        return df

class OutlierCleaner(object):

    def __init__(self,is_plot=True):
        self.is_plot=is_plot

    # def __plothist(self,factor):
    #     plt.hist(factor.values, bins=12, color='b', alpha=.4)

    def __Nstd(self,thisGroup,N=3):
        #N倍标准差，异常值
        thisM=thisGroup.mean(axis=1)
        thisS=thisGroup.std(axis=1)
        cap=thisM+N*thisS
        flo=thisM-N*thisS
        thisGroupC=thisGroup.copy()
        capM = pd.DataFrame(np.tile(np.mat(cap).T,thisGroup.shape[1]),index=thisGroup.index,columns=thisGroup.columns)
        floM = pd.DataFrame(np.tile(np.mat(flo).T,thisGroup.shape[1]),index=thisGroup.index,columns=thisGroup.columns)
        thisGroupC[thisGroup>cap]=capM
        thisGroupC[thisGroup<flo]=floM
        return thisGroupC

    def Nstd_pannel(self,factor):
        #N倍标准差，面板数据
        new_factor = self.__Nstd(factor,N=3)
        # if self.is_plot: self.__plothist(new_factor)
        return new_factor

class Standardization(object):

    def __init__(self):
        pass

    # 直接标准化
    def ZScore(self, thisGroup):
        thisM = thisGroup.mean(axis=1)
        thisS = thisGroup.std(axis=1)
        thisGroupC = thisGroup.copy()
        thisGroupC = (thisGroupC.sub(thisM.values,axis=0)).div(thisS.values,axis=0)
        return thisGroupC

    # 分行业进行zscore标准化
    def ZScore_Ind(self, factorM,industry):
        date = factorM.index
        stk = factorM.keys()
        result = pd.DataFrame(np.full_like(factorM,np.nan), index=date, columns=stk)

        for _ in range(1,np.max(industry.values)+1):
            df_ind = pd.DataFrame(np.full_like(factorM, np.nan), index=date, columns=stk)
            df_ind[industry==_] = factorM
            thisM = df_ind.mean(axis=1)
            thisS = df_ind.std(axis=1)
            thisGroupC = df_ind.copy()
            thisGroupC = (thisGroupC.sub(thisM.values, axis=0)).div(thisS.values, axis=0)
            result[thisGroupC!=np.nan]=thisGroupC

        return result

    # 分行业进行quantile标准化
    def QuantileChange_Ind(self, factorM,industry):
        date = factorM.index
        stk = factorM.keys()
        result = pd.DataFrame(np.full_like(factorM, np.nan), index=date, columns=stk)

        for _ in range(1,np.max(industry.values)+1):
            df_ind = pd.DataFrame(np.full_like(factorM, np.nan), index=date, columns=stk)
            df_ind[industry == _] = factorM
            df_rank = df_ind.rank(axis=1, method='dense', na_option='keep', ascending=True)
            thisM = df_rank.mean(axis=1)
            thisS = df_rank.std(axis=1)
            thisGroupC = df_rank.copy()
            thisGroupC = (thisGroupC.sub(thisM.values, axis=0)).div(thisS.values, axis=0)
            result[thisGroupC!=np.nan]=thisGroupC

        return result

    # 分行业进行quantile缩放至[0，1]区间内
    def QuantileMaxMin_Ind(self,factorM,industry):
        date = factorM.index
        stk = factorM.keys()
        result = pd.DataFrame(np.full_like(factorM, np.nan), index=date, columns=stk)

        for _ in range(1,np.max(industry.values)+1):
            df_ind = pd.DataFrame(np.full_like(factorM, np.nan), index=date, columns=stk)
            df_ind[industry == _] = factorM
            df_rank = df_ind.rank(axis=1, method='dense', na_option='keep', ascending=True)
            df_rank.fillna(0, inplace=True)
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train_minmax = min_max_scaler.fit_transform(df_rank.values.T)
            X_train_minmax = pd.DataFrame(X_train_minmax.T, index=date, columns=stk)
            result[X_train_minmax != 0] = X_train_minmax

        return result

class Orthogonalized(object):
    # 正交函数
    # 行业哑变量回归取残差
    def __init__(self):
        pass

    def get(self, factorM,industry):
        # data原始数据
        y = factorM.stack()
        x = industry.stack()
        data = pd.concat([x,y],axis=1)

        data.dropna(inplace=True)

        Resid = []
        for key, group in data.groupby(level=0,axis=0):
            tempResid = self.OLS(group)
            Resid.append(tempResid)
        Resid = pd.concat(Resid)
        Resid = Resid.unstack()
        return Resid

    def OLS(self, data):
        y = data.iloc[:,1]
        x = data.iloc[:,0]

        x = pd.get_dummies(x, prefix='industry')
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
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 定义中文字体
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
        xticklabels = Para.index[lable].astype(str).to_datetime().strftime('%Y-%m')
        axes2.set_xticks(lable)
        axes2.set_xticklabels(xticklabels, rotation=15)
        plt.grid(True, color='k', alpha=0.2)
        plt.title(u'因子系数f值时间序列', fontsize=15)

        plt.figure(3)
        fig3, axes3 = plt.subplots(1, figsize=(10, 6))
        Para['t'].abs().plot(kind='bar', color='b', alpha=0.7)
        plt.legend(['abs_tstat'], loc='upper right', fontsize=15)
        xticklabels = Para.index[lable].astype(str).to_datetime().strftime('%Y-%m')
        axes3.set_xticks(lable)
        axes3.set_xticklabels(xticklabels, rotation=15)
        plt.grid(True, color='k', alpha=0.2)
        plt.title(u'回归t值得绝对值', fontsize=15)
        plt.show()

    def OLS(self, data):
        y = data.loc[:,'futRet']
        x = data.loc[:, 'factor']
        results = sm.OLS(y, x).fit()
        f = results.params.iloc[0]
        t = results.tvalues.iloc[0]
        time = data.index[0][0]
        r = [time, f, t]
        return r

    def get(self, factorM,futRet, method='OLS', isSummary=True, isPlot=True):
        factor = factorM.stack()
        ret= futRet.stack()
        data = pd.concat([factor,ret],axis=1)
        data.columns = ['factor','futRet']
        data.dropna(inplace=True)
        assert ~(data.isnull().any().any()), '含nan报错'

        Para = []  # 记录检验值
        for key, group in data.groupby(level=0,axis=0):
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
        absmu = float(ic.rankIC.abs().mean())
        std = float(ic.rankIC.std())
        pct = float((ic.rankIC > 0).sum() / lenIC) * 100
        IR = float(mu / std)
        meanT = float(ic.t.abs().mean())
        tt = sum(ic.t.abs() > t) / lenIC
        t_pos = sum((ic.rankIC > 0) & (ic.t.abs() > t)) / sum(ic.rankIC > 0) * 100
        t_neg = sum((ic.rankIC < 0) & (ic.t.abs() > t)) / sum(ic.rankIC < 0) * 100
        x = np.array([mu * 100, absmu*100,std, pct, IR, meanT, tt, t_pos, t_neg])
        result = pd.DataFrame({'Value': np.round(x, 2),
                               'Name': ['IC_mean', 'IC_abs_mean','IC_std', 'IC_pos_per', 'IR', 't_mean', 'sign_all', 'sign_pos',
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
        xticklabels = rankIC.index[lable].astype(str).to_datetime().strftime('%Y-%m')
        xtick = lable
        axes.set_xticks(xtick)
        axes.set_xticklabels(xticklabels, rotation=30, fontsize=15)
        plt.grid(True, color='k', alpha=0.2)
        plt.show()

    def rankIC(self, factorM,futRet, t=1.96, isSummary=True, isPlot=True,isShift=True):
        if isShift: futRet = futRet.shift(1)
        factor = factorM.stack()
        ret= futRet.stack()
        data = pd.concat([factor,ret],axis=1)
        data.columns = ['factor','futRet']
        data.dropna(inplace=True)
        assert ~(data.isnull().any().any()), '含nan报错'

        ic = data.groupby(level=0,axis=0).agg(lambda x: np.array(ss.spearmanr(x)))
        ic.columns = ['rankIC', 'pValue']
        rs = ic.loc[:, 'rankIC'].values
        ic.loc[:, 't'] = rs * np.sqrt((len(data) / len(rs) - 2) / ((rs + 1.0) * (1.0 - rs)))

        summary = self.__summary(ic, t)
        if isSummary:
            print(summary)
        if isPlot:
            self.__draw(ic)

        ic.unstack()
        return ic, summary


class SpiltGroup(object):
    # 分组
    def __init__(self):
        pass

    def __GroupInd(self, lenG):
        # 分组的序号
        if lenG == 1:
            GroupInd3 = np.tile([0, 1],(self.GroupNum,1))
        elif lenG == 2:
            GroupInd3 = np.array(math.floor(float(self.GroupNum) / 2.) * [[0, 1]] + (self.GroupNum - math.floor(float(self.GroupNum) / 2.)) * [[1, 2]])
        elif lenG < self.GroupNum:
            GroupInd = np.arange(lenG)
            y = self.GroupNum - lenG
            GroupInd2 = [[i, i + 1] for i in GroupInd]
            GroupInd2.extend(y * [GroupInd2[-1]])
            if len(GroupInd2) != self.GroupNum: raise Exception
            GroupInd3 = np.array(GroupInd2)
        else:
            GroupInd = np.arange(0, lenG, np.floor(lenG / self.GroupNum))
            GroupInd2 = [[GroupInd[i], (GroupInd[i + 1] if (i + 1) <= len(GroupInd) - 1 else lenG)] for i in
                         np.arange(self.GroupNum)]
            GroupInd3 = np.array(GroupInd2)
            GroupInd3[self.GroupNum - 1, 1] = lenG
        GroupInd3 = GroupInd3.astype(int)
        if len(GroupInd3) != self.GroupNum: raise Exception
        return GroupInd3

    def __Group_Simple_AVG(self, data):
        # 不考虑行业，简单的平均分组
        # 按照主要因子排序
        data = data.sort_values(by=self.main, ascending=not (self.isDesc), inplace=False)
        # 分组
        GroupInd = self.__GroupInd(data.shape[0])

        # 循环每一组
        for i in np.arange(self.GroupNum):
            # 分组序号
            indexloc = data.index[GroupInd[i, 0]:GroupInd[i, 1]]

            # 组号
            data.loc[indexloc, 'group'] = i
            # 分配权重
            if self.WeightType == 'Simple_AVG':
                data.loc[indexloc, 'weight'] = 1.00 / len(indexloc)
            elif self.WeightType == 'MarketValue':
                # 市值加权，输入的data应该有MV列
                data.loc[indexloc, 'weight'] = (data.loc[indexloc, 'marketValue'].values) / data.loc[indexloc, 'marketValue'].sum()
            else:
                raise Exception  # 注意，简单分组里面不能用大行业

        return data

    def __Group_Industry_AVG(self, data,industryM):
        # 按照指数内的行业数量平均分组，当行业内的股票数量小于分组的时候，先顾头尾，
        # data必须包含Industry和MV列和WEIGHT列
        # 权重按指数成分进行划分
        if self.isIndustryNeutral is True:
            uniDataIndustry = pd.DataFrame(pd.unique(data['industry']), columns=['industry'])
            industryWeight = self.__Weight_IndustryNeutralWeight(uniDataIndustry, thisdate)
            data = pd.merge(data, industryWeight, on='industry', how='left')
            if data['industryWeight'].isnull().any().any(): raise Exception
        else:
            data['industryWeight'] = 1

        data['denominator'] = np.nan  # 权重分母
        # 分组的组内list初始化
        groupList = [[] for i in np.arange(self.GroupNum)]
        # 先按照行业分组，计算权重，并放入不同的组内
        for key, group in data.groupby(by='industry'):
            tempdata = group.sort_values(by=self.main, ascending=not (self.isDesc), inplace=False).copy()  # 按照主要因子排序
            GroupInd = self.__GroupInd(tempdata.shape[0])  # 一个行业五组分组序号
            for i in np.arange(self.GroupNum):  # 行业内分组
                tempdata2 = tempdata.iloc[GroupInd[i, 0]:GroupInd[i, 1], :].copy()
                # 行业内权重分母赋值
                if self.WeightType == 'Simple_AVG':
                    tempdata2.loc[:, 'denominator'] = len(tempdata2)
                elif self.WeightType == 'MarketValue':
                    tempdata2.loc[:, 'denominator'] = sum(tempdata2.loc[:, 'marketValue'].values)
                # 标明组号
                tempdata2.loc[:, 'group'] = i
                groupList[i].append(tempdata2)

        # 整理分好组的部分
        groupListDF = []
        for i in np.arange(self.GroupNum):
            thisDF = pd.concat(groupList[i])
            # 计算组内权重
            if self.WeightType == 'Simple_AVG':
                thisDF.loc[:, 'weight'] = 1 / thisDF.loc[:, 'denominator'].values * thisDF.loc[:,
                                                                                    'industryWeight'].values
            elif self.WeightType == 'MarketValue':
                thisDF.loc[:, 'weight'] = thisDF.loc[:, 'marketValue'] / thisDF.loc[:,
                                                                         'denominator'].values * thisDF.loc[:,
                                                                                                 'industryWeight'].values

            groupListDF.append(thisDF)

        result = pd.concat(groupListDF)
        return result

    def __Weight_IndustryNeutralWeight(self, uniDataIndustry, thisdate):
        # 前提是，分组按行业来分，大行业的权重，行业内部可以是简单平均或者流通市值分组
        # 注意，由于拿到的指数行业是延迟的，所以可能出现两种特殊情况，即：
        # 1.指数内有某行业，但组中没有，此时去掉这些行业，重新计算每个行业的权重。注意，这里如果被抛弃的行业权重超过5%，将停止检查
        # 2.组内有某行业，但指数中没有，那么把这个行业纳入进来，但是权重为0
        uniDataIndustry.loc[:, 'value'] = 1
        if self.IndexIndustryWeight is None: raise Exception
        thisIndexIndustryWeight = self.IndexIndustryWeight.loc[
            self.IndexIndustryWeight.loc[:, 'date'] == thisdate, ['industry', 'industryWeight']]
        #        IndexIndustry=thisIndexIndustryWeight.loc[:,'Industry']
        mergedata = pd.merge(thisIndexIndustryWeight, uniDataIndustry, how='outer')
        if mergedata.isnull().any().any():
            # 当日行业权重为空，或者行业多出来
            #        if (len(mergedata)!=len(thisIndexIndustryWeight)) | (len(uniDataIndustry)!=len(thisIndexIndustryWeight)):
            #            raise #这段没实际应用过
            mergedata.loc[mergedata['industryWeight'].isnull(), ['industryWeight']] = 0
            mergedata.dropna(0, how='any', inplace=True)
            if sum(mergedata['industryWeight']) <= 0.95: raise Exception
            mergedata.loc[:, 'industryWeight'] = mergedata.loc[:, 'industryWeight'].values / mergedata.loc[:,
                                                                                             'industryWeight'].sum()

        mergedata = mergedata.loc[:, ['industry', 'industryWeight']]
        if mergedata.loc[:, 'industryWeight'].isnull().any(): raise Exception
        return mergedata

    '''     
    data为数据源，main为分组中的主要因子，用这个因子来排序和分组
    minor为次要因子，用来探索次要因子相对于主要因子的单调性，isDesc为因子方向，默认是【降序】排列
    若minor次要因子为超额收益率，那么可以使needRet为True，从而计算收益净值曲线
    GroupNum为分组数，默认为5，也可以设置为10；
    SpiltGroupType为分组方式选择，默认为简单平均分组，可选择，按行业数量平均分配到各组
    WeightType为分组后的权重分配。默认为简单平均分配权重
    isIndustryNeutral为是否使用行业中性分配大行业的权重，注意，行业中性仅在使用行业分组中
    使用行业中性权重时，需要将IndexIndustryWeight赋值
    isplot控制是否画图，默认为True画图
    '''
    def get(self, factorM,futRet,industryM=None,zz500=None,ffc=None,main='factor', isDesc=True, minor='ret', needRet=False, GroupNum=5,
            SpiltGroupType='Simple_AVG', WeightType='Simple_AVG', isIndustryNeutral=False, isplot=True,isShift=True):

        if isShift: futRet = futRet.shift(1)
        factor = factorM.stack()
        ret= futRet.stack()
        industry = industryM.stack()

        print('分组计算开始')
        if main == 'factor':
            print('主要因子默认为factor')
        if minor is None:
            print('次要因子或收益率需注明')
            raise (IOError)
        if isIndustryNeutral:
            print('权重中性化:导入指数行业权重')
            zz500Weight = zz500.stack()
            FFCap = ffc.stack()
            data = pd.concat([factor, ret, industry, zz500Weight,FFCap], axis=1)
            data.columns = ['factor', 'ret', 'industry', 'weight','marketValue']
            data.dropna(inplace=True)
            for i in ['group', 'cumret']:  # 组号，权重，收益率
                data[i] = np.nan  # 组号
        elif ~isIndustryNeutral & (WeightType == 'MarketValue'):
            print('流通市值加权，导入流通市值')
            FFCap = ffc.stack()
            data = pd.concat([factor, ret, industry,FFCap], axis=1)
            data.columns = ['factor', 'ret', 'industry', 'marketValue']
            data.dropna(inplace=True)
            for i in ['group', 'weight', 'cumret']:  # 组号，权重，收益率
                data[i] = np.nan  # 组号
        else:
            print('简单分组加权')
            data = pd.concat([factor, ret, industry], axis=1)
            data.columns = ['factor', 'ret', 'industry']
            data.dropna(inplace=True)
            for i in ['group', 'weight', 'cumret']:  # 组号，权重，收益率
                data[i] = np.nan  # 组号

        dataOri = data.copy()
        # 赋值
        self.isDesc, self.GroupNum, self.main, self.minor, self.WeightType, self.needRet, self.isIndustryNeutral = isDesc, GroupNum, main, minor, WeightType, needRet, isIndustryNeutral
        if self.isIndustryNeutral: self.IndexIndustryWeight=zz500Weight

        # 按日期循环
        newdata = []
        if SpiltGroupType == 'Simple_AVG':
            for key, group in data.groupby(level=0,axis=0):
                result = self.__Group_Simple_AVG(group)
                newdata.append(result)
            newdata = pd.concat(newdata)

        elif SpiltGroupType == 'Industry_AVG':
            newdata = self.__Group_Industry_AVG(data,industryM)

        # 分组后新数据拼接
        newdata.reset_index(drop=True, inplace=True)
        newdata = newdata.loc[:, ['date', 'code', 'group', 'weight']]

        data = pd.merge(newdata, dataOri.reset_index(), how='left', on=['code', 'date'])
        meanMain = data.groupby(['group', 'date'])[self.main].mean()
        meanMain = meanMain.unstack('group')

        meanMinor = data.groupby(['group', 'date'])[self.minor].mean()
        meanMinor = meanMinor.unstack('group')

        turnoverRate = self.turnoverRate(newdata)
        result = {'groupData': newdata, 'mono': meanMinor, 'turnoverRate': turnoverRate}

        # 单调性画图
        if isplot:
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 2, 1)
            ax21 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            plt.grid(True)  # 添加网格
            plt.ion()  # interactive mode on

            ax1.title.set_text('Stabilization of Main Factor ')
            ax21.title.set_text('Monotonicity of Minor Factor')
            ax3.title.set_text('Turnover Rate')
            ax4.title.set_text('Net Value of Different Group ')

            for i in np.arange(GroupNum): ax1.plot(np.arange(len(meanMain)),
                                                   meanMain[i].T.values)  # ,'',label="Group"+str(i+1))
            #            ax1.legend(loc = 'upper left', fontsize=15)

            barwidth = 0.35
            ax21.bar(np.arange(GroupNum), meanMain.mean(axis=0), barwidth, color='b', alpha=0.7)
            ax22 = ax21.twinx()  # this is the important function
            ax22.bar(np.arange(GroupNum) + barwidth, meanMinor.mean(axis=0), barwidth, color='r', alpha=0.7)
            self.__align_yaxis(ax21, ax22, )
            #            plt.legend([r'$\mu_{group}$'], loc='upper right',fontsize=15)

            ax3.bar(np.arange(GroupNum), turnoverRate.mean(axis=0))
            ax3.legend(loc='upper left', fontsize=15)

        if self.needRet:
            data.loc[:, 'w_ret'] = (1 + data.loc[:, self.minor].values) * data.weight
            sumRet = data.groupby(['group', 'date']).w_ret.sum()
            sumRet = sumRet.unstack('group')

            netV = sumRet.cumprod(axis=0)  # 粗略分组净值
            # 需要画图:均值--bar，sumRet--plot彩色
            if isplot:
                plotData = pd.DataFrame(netV, index=np.unique(data['date']))
                for i in np.arange(GroupNum): ax4.plot(np.arange(len(plotData)), plotData[i].T.values,
                                                       'C' + str(int(i)), label="Group" + str(i + 1))
                tick = np.arange(0, len(plotData), 12)
                label = plotData.index[tick].strftime('%Y-%m')
                label = label.tolist()
                ax4.legend(loc='upper left', fontsize=15)
                ax4.set_xticks(tick)
                ax4.set_xticklabels(label, rotation=40)

                ax4.set_xlabel('Date')
                ax4.set_ylabel('Net Value')
                plt.show()

        return result

    def __align_yaxis(self, ax1, ax2):
        # 画图专用
        """Align zeros of the two axes, zooming them out by same ratio"""
        axes = (ax1, ax2)
        extrema = [ax.get_ylim() for ax in axes]
        tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
        # Ensure that plots (intervals) are ordered bottom to top:
        if tops[0] > tops[1]:
            axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]
        # How much would the plot overflow if we kept current zoom levels?
        tot_span = tops[1] + 1 - tops[0]

        b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
        t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
        axes[0].set_ylim(extrema[0][0], b_new_t)
        axes[1].set_ylim(t_new_b, extrema[1][1])

    def turnoverRate(self, groupData):
        rateList = []
        for key1, data1 in groupData.groupby('group'):
            lastCode = []
            for key2, data2 in data1.groupby('date'):
                nowCode = data2.code.tolist()
                if len(lastCode) > 0:
                    rate = len(set(lastCode) - set(nowCode)) / len(lastCode)
                    rateList.append([key1, key2, rate])
                else:
                    rateList.append([key1, key2, np.nan])
                lastCode = nowCode.copy()

        rateList = pd.DataFrame(rateList, columns=['group', 'date', 'rate'])
        rateList2 = rateList.set_index(['date', 'group'])
        rateList3 = rateList2.unstack('group')
        return rateList3

    def GroupToPortfolio(self, data, GroupWeight=[1]):
        # GroupWeight是组比重的list
        # 如果GroupWeight没给参数，那就是1份第一组
        if type(GroupWeight) != list: raise Exception
        if (len(GroupWeight) == 1) & (GroupWeight[0] != 1): raise Exception
        Portfolio = []
        data.reset_index(inplace=True)
        for i in np.arange(len(GroupWeight)):
            thisPortfolio = data.loc[data['group'] == i, ['code', 'date', 'weight']].copy()
            thisPortfolio.loc[:, 'weight'] = thisPortfolio.loc[:, 'weight'] * GroupWeight[i]
            Portfolio.append(thisPortfolio)
        Portfolio = pd.concat(Portfolio)
        if len(GroupWeight) > 1:
            Portfolio = Portfolio.groupby(by=['code', 'date']).sum()
        Portfolio.reset_index(inplace=True)
        Portfolio = Portfolio.loc[Portfolio.weight != 0, ['code', 'date', 'weight']]
        return Portfolio