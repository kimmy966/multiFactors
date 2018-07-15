import pandas as pd
import statsmodels.formula.api as sm

class neutral_process(object):

    # 对多风格因子进行回归，取残差

    def __init__(self):
        pass

    def get(self, data,Y_columns, X_columns):
        # data原始数据
        if type(Y_columns) is not list: Y_columns = [Y_columns]
        if type(X_columns) is not list: X_columns = [X_columns]
        data.dropna(inplace=True)

        self.yc, self.xc = Y_columns, X_columns

        Resid = []
        for key, group in data.groupby(level=0,axis=0):
            tempResid = self.OLS(group)
            Resid.append(tempResid)
        Resid = pd.concat(Resid)
        Resid = Resid.unstack()
        return Resid

    def OLS(self, data):
        y = data.loc[:, self.yc]
        x = data.loc[:, self.xc]

        x = pd.get_dummies(x, prefix='industry')
        results = sm.OLS(y, x).fit()
        r = pd.DataFrame(results.resid, columns=['factor'], index=data.index)
        return r