#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn import linear_model

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#naivebayes not for regresion
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso,Ridge

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
#import xgboost as xgb
from xgboost import XGBRegressor
import xgboost
from sklearn.model_selection import cross_val_score

#pip install xgboost

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore",category=FutureWarning)

data = pd.read_excel("CapstoneProject.xlsx")  # predict strength of cement. Regression problem

# ## Exploratory Data Analysis

def custom_summary(dataframe):
    from collections import OrderedDict    # allows user to create a dictionary giving better options compared to other dictionary library
    result = []
    for col in list(dataframe.columns):    # conversion ofarray to list since list management is easy
        stat = OrderedDict({'Feature_Name' : col ,
                           'Count' : dataframe[col].count(),
                           'Data_type' : dataframe[col].dtype,
                           'Minimum' : dataframe[col].min(),
                           'Quartile_1' : dataframe[col].quantile(0.25),
                           'Mean' : dataframe[col].mean(),
                           'Median' : dataframe[col].median(),
                           'Quartile_3' : dataframe[col].quantile(0.75),
                           'Maximun' : dataframe[col].max(),
                           'Standard_Deviation' : dataframe[col].std(),
                           'Kurtosis' : dataframe[col].kurt(),
                           'Skewness' : dataframe[col].skew(),
                           'Range' : dataframe[col].max()-dataframe[col].min(),
                           'InterQuartile_Range' :dataframe[col].quantile(0.75)-dataframe[col].quantile(0.25)
                           })
        result.append(stat)  #append is a list function
        # Adding skewness comment
        if dataframe[col].skew() <= -1:
            sklabel = 'High Negative Skew'
        elif -1 <= dataframe[col].skew() < -0.5:
            sklabel = 'Moderate Negative Skew'
        elif -0.5 <= dataframe[col].skew() < 0:
            sklabel = 'Fairly Symmetric (Negative)'
        elif 0 <= dataframe[col].skew() < 0.5:
            sklabel = 'Fairly Symmetric (Positive)'
        elif 0.5 <= dataframe[col].skew() < 1:
            sklabel = 'Moderate Positive Skew'
        elif dataframe[col].skew() >= 1:
            sklabel = 'High Positive Skew'
        else:
            sklabel = 'Error!'
        stat['Skewness_Comment'] = sklabel
            
        # Adding Outlier Function
        Upper_Limit = stat['Quartile_3'] + (1.5*stat['InterQuartile_Range'])
        Lower_Limit = stat['Quartile_1'] - (1.5*stat['InterQuartile_Range'])
        if len([x for x in dataframe[col] if x < Lower_Limit or x > Upper_Limit]) > 0:
            OutLabel = 'Has Outlier'
        else:
            OutLabel = 'No Outlier'
        stat['Outlier_Comment'] = OutLabel
    result_df = pd.DataFrame(data = result)  # create a dataframe using pd.DataFrame function
    #return result_df.T                       #Transpose the output to get column names in the form of 'columns'.If many variables no need to take transpose.
    return result_df

custom_summary(data)


sns.distplot(data['cement'])

sns.distplot(data['slag'])

sns.distplot(data['strength'])

def ODT_Plots(dataframe,col):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize = (20,6))
    kwargs = {'Fontsize' : 20,'color' : 'black'}    # keyword args: dictionary which will be replaced when called
    # BoxPlot distrbution
    sns.boxplot(dataframe[col], ax = ax1, color = 'g', orient = "v")
    ax1.set_title('Boxplot for '+col , **kwargs)
    ax1.set_xlabel('Values', **kwargs)
    ax1.set_ylabel('Boxplot Distribution', **kwargs)
    
    # Histogram with outliers
    sns.distplot(dataframe[col],ax = ax2, color = 'r',fit = stats.norm)  #adds normal distri curve to grapg using mean +/- std method
    ax2.set_title('Histplot with outliers for '+col , **kwargs)   
    ax2.set_xlabel('Distribution', **kwargs)
    ax2.set_ylabel('Values', **kwargs)
    ax2.axvline(dataframe[col].mean(),color = 'g',linestyle = '--')  # -- or dashed
    ax2.axvline(dataframe[col].median(),color = 'black')
    
for col in list(data.columns):
    ODT_Plots(data,col)


# ## Outlier Detection

def otf(dataf, col, method= 'quartile', strategy= 'median'):
    coldata= dataf[col]  # cretes a list
    
    if method== 'quartile':
        colmedian= dataf[col].median()
        q3= dataf[col].quantile(.75)
        q1= dataf[col].quantile(0.25)
        IQR = q3 - q1
        upper_limit= q3 + 1.5*IQR
        lower_limit= q1 - 1.5*IQR
    elif method== 'std':
        colmean= dataf[col].mean()
        colstd= dataf[col].std()
        cutofff= colstd*2 
        upper_limit= colmean+ cutofff
        lower_limit= colmean- cutofff
    else:
        print('Error: Select a valid method from quartile or std')
    outliers= dataf.loc[(coldata<lower_limit) | (coldata>upper_limit), col]
    outlier_density= round(len(outliers)/len(dataf)*100,2)
    if outlier_density == 0:
        print(f'feature \"{col}\" does not have any outlier')
    else:
        print(f" Total number of outliers are: {len(outliers)}\n")
        print(f"outlier density is : {outlier_density}% \n")
        print(f" outliers for \'{col}\' are: \n {np.array(outliers)} \n")
        display(dataf[(coldata<lower_limit)|(coldata>upper_limit)])
        
    
    if strategy == 'median':
        dataf.loc[(coldata<lower_limit)|(coldata>upper_limit), col] = colmedian
    elif strategy == 'mean':
        dataf.loc[(coldata<lower_limit)|(coldata>upper_limit), col] = colmean
    
    else:
         print(f"Feature \"{col}\" does not need outlier treatment")
    return dataf

for col in list(data.columns):
    otf(data,col)

# ## Checking for Multicollinearity

data.corr()

def vif_check(independent_var):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_result = pd.DataFrame()
    vif_result['vif_factor'] = [variance_inflation_factor(independent_var.values,i) for i in range(independent_var.shape[1])]
    vif_result['feature'] = independent_var.columns
    return vif_result.sort_values('vif_factor',ascending = False)

vif_check(data.drop('strength',axis=1))  # shows r2 score  . threshold is 5.not just here

# ## Treating Multicollinearity using PCA

def apply_pca(x):   # data excluding the target column
    col = []    #append transformed columns
    n_component = len(x.columns)     
    
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)      # pca done on the entire data (train + test). when scaled: fit-transform on entire data
    from sklearn.decomposition import PCA 
    
    for i in range(1,n_component + 1):   # for no.of columns in data
        pca = PCA(n_components = i)  #no.of pca increase accordingly
        p_components = pca.fit_transform(x)
        total_explained_variance = np.cumsum(pca.explained_variance_ratio_)   # gives a list(hence next line has i-1)
        if total_explained_variance[i-1] > 0.9:
            n_components = i
            break
                      
    print('the total explained variance ratio is: ',total_explained_variance)
    
    for j in range(1,n_components + 1):
        col.append('pc' + str(j))        # col name + col number
    
    result_df = pd.DataFrame(data = p_components, columns = col)
    return result_df

data_copy1 = data.copy()   

data_with_pca = apply_pca(data.drop('strength',axis = 1))

new_df = data_with_pca.join(data[['strength']],how='left')
new_df

# ## Model building
# train test split

def train_and_test_split(data,target_col,test_size = 0.3):
    x = data.drop(target_col,axis=1)
    y = data[target_col]
    return train_test_split(x,y,test_size=test_size,random_state = 100)    

def build_model(model_name,estimator,data,target_col):
    x_train,x_test,y_train,y_test = train_and_test_split(data,target_col)
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    r2_score_val = r2_score(y_test,y_pred)
    result = [model_name,rmse,mae,r2_score_val]
    return result    


# ## Model Ver.1

build_model(model_name='LinearRegression',estimator=LinearRegression(),data=data_copy1,target_col='strength')

# ## Model Ver.2

build_model(model_name='LinearRegression',estimator=LinearRegression(),data=new_df,target_col='strength')

# Trying multiple models 

def multiple_models(data_n, target_n):  
    col_names = ['model_name','rmse','mae','r2_score_value']
    result = pd.DataFrame(columns=col_names)
    result.loc[len(result)]  = build_model('LinearRegression',estimator=LinearRegression(),data=data_n,target_col=target_n)  # result of loc0 0
    result.loc[len(result)]  = build_model('LassoRegression',estimator=Lasso(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('RidgeRegression',estimator=Ridge(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('DecisionTree',estimator=DecisionTreeRegressor(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('SupportVector',estimator=SVR(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('KNeighbors',estimator=KNeighborsRegressor(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('RandomForest',estimator=RandomForestRegressor(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('AdaBoost',estimator=AdaBoostRegressor(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('Gboost',estimator=GradientBoostingRegressor(),data=data_n,target_col=target_n)
    result.loc[len(result)]  = build_model('XGBoost',estimator=XGBRegressor(),data=data_n,target_col=target_n)
    
    return result       

# ## Model Ver1

multiple_models(data,'strength')

# ## Model Ver2

multiple_models(new_df,'strength')

# ## Applying Cross Validation

def k_Fold_CV(x,y,fold = 10):
    
    score_lr = cross_val_score(LinearRegression(),x,y,cv=fold)
    score_lasso = cross_val_score(Lasso(),x,y,cv=fold)
    score_ridge = cross_val_score(Ridge(),x,y,cv=fold)
    score_DT = cross_val_score(DecisionTreeRegressor(),x,y,cv=fold)
    score_SVR = cross_val_score(SVR(),x,y,cv=fold)
    score_KN = cross_val_score(KNeighborsRegressor(),x,y,cv=fold)
    score_RF = cross_val_score(RandomForestRegressor(),x,y,cv=fold)
    score_AB = cross_val_score(AdaBoostRegressor(),x,y,cv=fold)
    score_GB = cross_val_score(GradientBoostingRegressor(),x,y,cv=fold)
    score_XGB = cross_val_score(XGBRegressor(),x,y,cv=fold)
    
    estimators = [LinearRegression(),Lasso(),Ridge(),DecisionTreeRegressor(),SVR(),KNeighborsRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),XGBRegressor()]
    
    estimators_name = ['LinearRegression','Lasso','RidgeRegression','DecisionTree','SupportVector','KNeighbors','RandomForest','AdaBoost','GBoost','XGBoost']
    
    score_list = [score_lr,score_lasso,score_ridge,score_DT,score_SVR,score_KN,score_RF,score_AB,score_GB,score_XGB]
    
    result =[]
    
    for i in range(0,len(estimators)):
        score_mean = np.mean(score_list[i])
        score_std = np.std(score_list[i])
        names = estimators_name[i]
        temp = [names , score_mean , score_std]
        result.append(temp)
    
    result_df = pd.DataFrame(data = result, columns = ['model_names','mean_scores','std_scores'])
    return result_df    

x_pre = data.drop('strength',axis=1)

y_pre = data[['strength']]

x_post = new_df.drop('strength',axis=1)

y_post = new_df[['strength']]

# ## Model ver1 using cross validation

k_Fold_CV(x_pre,y_pre)

# ## Model ver2 using cross validation

k_Fold_CV(x_post,y_post)  # pca treated data

# ## Hyperparameter tuning

def parameter_tune(x,y,fold=10):
    dt = DecisionTreeRegressor()
    rf = RandomForestRegressor()
    ab = AdaBoostRegressor()
    gb = GradientBoostingRegressor()
    xgb = XGBRegressor()
    knn = KNeighborsRegressor()
    
    # defining parameter grids for all models,   # seq does not matter
    param_dt = {'max_depth' : [5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]}  # model run for each depth
    param_rf = {'n_estimators' : [50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500]}
    param_ab = {'n_estimators':[50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500],'learning_rate':[0.1,1]}
    param_gb = {'alpha' : [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]}    # loss function
    param_xgb = {'reg_lambda' : [0,1]}
    param_knn = {'n_neighbors' : [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]}
    
    
    # fit in gridsearchcv
    tune_dt = GridSearchCV(dt,param_dt,cv = fold)
    tune_rf = GridSearchCV(rf,param_rf,cv = fold)
    tune_ab = GridSearchCV(ab,param_ab,cv = fold)
    tune_gb = GridSearchCV(gb,param_gb,cv = fold)
    tune_xgb = GridSearchCV(xgb,param_xgb,cv = fold)
    tune_knn = GridSearchCV(knn,param_knn,cv = fold)
    
    tune_dt.fit(x,y)
    tune_rf.fit(x,y)
    tune_ab.fit(x,y)
    tune_gb.fit(x,y)
    tune_xgb.fit(x,y)
    tune_knn.fit(x,y)
    
    
    tune_models = [tune_dt,tune_rf,tune_ab,tune_gb,tune_xgb,tune_knn]
    models = ['DecisionTree','RandomForest','AdaBoost','GradientBoosting','XGB','KNN']
    
    
    for i in range(0,len(tune_models)):
        print('Model Name',models[i])
        print('Best Parameter',tune_models[i].best_params_)
        
parameter_tune(x_post,y_post)  # giving x and y after doing PCA

def k_Fold_CV_hyperTuned(x,y,fold = 10):
    
    score_lr = cross_val_score(LinearRegression(),x,y,cv=fold)
    score_lasso = cross_val_score(Lasso(),x,y,cv=fold)
    score_ridge = cross_val_score(Ridge(),x,y,cv=fold)
    score_DT = cross_val_score(DecisionTreeRegressor(max_depth= 25),x,y,cv=fold)
    score_SVR = cross_val_score(SVR(),x,y,cv=fold)
    score_KN = cross_val_score(KNeighborsRegressor(n_neighbors= 4),x,y,cv=fold)
    score_RF = cross_val_score(RandomForestRegressor(n_estimators= 340),x,y,cv=fold)
    score_AB = cross_val_score(AdaBoostRegressor(learning_rate= 1, n_estimators= 80),x,y,cv=fold)
    score_GB = cross_val_score(GradientBoostingRegressor(alpha= 0.8),x,y,cv=fold)
    score_XGB = cross_val_score(XGBRegressor(reg_lambda= 1),x,y,cv=fold)
    
    estimators = [LinearRegression(),Lasso(),Ridge(),DecisionTreeRegressor(),SVR(),KNeighborsRegressor(),RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),XGBRegressor()]
    
    estimators_name = ['LinearRegression','Lasso','RidgeRegression','DecisionTree','SupportVector','KNeighbors','RandomForest','AdaBoost','GBoost','XGBoost']
    
    score_list = [score_lr,score_lasso,score_ridge,score_DT,score_SVR,score_KN,score_RF,score_AB,score_GB,score_XGB]
    
    result =[]
    
    for i in range(0,len(estimators)):
        score_mean = np.mean(score_list[i])
        score_std = np.std(score_list[i])
        names = estimators_name[i]
        temp = [names , score_mean , score_std]
        result.append(temp)
    
    result_df = pd.DataFrame(data = result, columns = ['model_names','mean_scores','std_scores'])
    return result_df

k_Fold_CV_hyperTuned(x_post,y_post)

k_Fold_CV(x_post,y_post)

# ## Feature Importance

x_train,x_test,y_train,y_test = train_test_split(x_pre,y_pre,test_size=0.3)

xgb = XGBRegressor()

xgb.fit(x_train,y_train)

xgboost.plot_importance(xgb)   # f score gives the importance of each feature. these features can be used to build other models as well

x1 = x_pre[['age','cement','water','coarseagg','fineagg','slag']]

k_Fold_CV_hyperTuned(x1,y_pre)

# ## Learning curve analysis

def generate_learningCurve(model_name,estimator,x,y):
    from sklearn.model_selection import learning_curve
    train_size,train_score,test_score = learning_curve(estimator = estimator,X=x,y=y,cv=10)
    train_score_mean = np.mean(train_score,axis = 1)  #axis=1 to access rows
    test_score_mean = np.mean(test_score,axis = 1)
    
    plt.plot(train_size,train_score_mean,color = 'blue')
    plt.plot(train_size,test_score_mean,color = 'green')
    plt.xlabel('samples')
    plt.ylabel('accuracy score')
    plt.title(model_name)
    plt.legend(('Train','Test'))

generate_learningCurve('LinearRegression',LinearRegression(),x1,y_pre)

from sklearn.model_selection import learning_curve

train_size,train_score,test_score = learning_curve(estimator = LinearRegression(),X=x1,y=y_pre,cv=10)

print(train_score)
