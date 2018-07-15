import singleFactorTest as sf
import pandas as pd

def read_data():
    factor1 = pd.read_csv('factors1.csv',index_col=0)
    industry = pd.read_csv('industryCiticsM.csv',index_col=0)
    futRet = pd.read_csv('futRet.csv',index_col=0)
    zz500 = pd.read_csv('zz500.csv',index_col=0)
    ffc = pd.read_csv('freeFloatCapM.csv',index_col=0)
    df= sf.DataDealing()
    industryM = df.dealing(factor1,industry)
    futRet = df.dealing(factor1,futRet)
    zz500 = df.dealing(factor1,zz500)
    ffc = df.dealing(factor1,ffc)
    return  factor1,futRet,industryM,zz500,ffc

# 标准化方法1：去异常值，ZScore法
def ZScore(ind_choice=True):
    factor1, futRet, industryM,_,_ = read_data()
    sf1 = sf.OutlierCleaner()
    sf1.Nstd_pannel(factor1)
    sf2 = sf.Standardization()
    if ind_choice: factor = sf2.ZScore(factor1)
    else: factor = sf2.ZScore_Ind(factor1,industryM)
    return factor

# 标准化方法2：直接取rank标准化
def Quantile(method_choice='maxmin'):
    factor1, futRet, industryM,_,_ = read_data()
    sf2 = sf.Standardization()
    if method_choice == 'sta': factor = sf2.QuantileChange_Ind(factor1,industryM)
    else: factor = sf2.QuantileMaxMin_Ind(factor1,industryM)
    return factor

# 因子行业中性处理
def ind_deal(method='Quantile'):
    if method=='Quantile':factor=Quantile()
    else: factor = ZScore()
    sf3 = sf.Orthogonalized()
    _,_,industryM,_,_ = read_data()
    factor = sf3.get(factor,industryM)
    return factor

# 回归计算序列t值
def tstats():
    factor = ind_deal()
    _,futRet,_,_ ,_= read_data()
    sf4 = sf.Regression()
    Para = sf4.get(factor,futRet)
    return Para

def cacu_rankIC():
    factor = ind_deal()
    _,futRet,_,_,_ = read_data()
    sf5 = sf.IC()
    ic, summary = sf5.rankIC(factor,futRet)
    return ic, summary

def split_test():
    factor, futRet, industryM, zz500,ffc = read_data()
    sf6 = sf.SpiltGroup()
    sf6.get(factor,futRet,industryM=industryM, zz500=zz500,ffc=ffc,isIndustryNeutral=True,SpiltGroupType = 'Industry_AVG')
    return

split_test()