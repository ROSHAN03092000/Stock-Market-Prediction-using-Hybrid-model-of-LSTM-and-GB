# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:02:32 2023

@author: rosha
"""
import pandas as pd
import numpy as np
name_stock=input("Enter a Stock name [BAJ,REL,SBI,TCS,MRF]:")
import matplotlib.pyplot as plt
plt.style.use("Solarize_Light2")
#value=int(input(f"Enter Rupees to be invested in {name_stock} :"))
#Creating a function to read csv file
def read_csv(name):
    if name=="BAJ":
        df=pd.read_csv("BAJFINANCE.NS.csv",parse_dates=['Date'])
    elif name=="TCS":
        df=pd.read_csv("TCS.NS.csv",parse_dates=['Date'])
    elif name=="SBI":
        df=pd.read_csv("SBIN.NS.csv",parse_dates=['Date'])
    elif name=="MRF":
        df=pd.read_csv("MRF.NS (1).csv",parse_dates=['Date'])
    elif name=="REL":
        df=pd.read_csv("RELIANCE.NS.csv",parse_dates=['Date'])
    df=df[(df['Date']<"2024-01-01")&(df['Date']>="2020-01-01")]
    return df
df=read_csv(name_stock)
#print(df.head())
df.set_index('Date',inplace=True,drop=False)
#print(df.to_string())
print(df.info())
print(df.describe())
print(df.duplicated())
BAJ=pd.read_csv("BAJFINANCE.NS.csv",parse_dates=['Date'])
TCS=pd.read_csv("TCS.NS.csv",parse_dates=['Date'])
SBI=pd.read_csv("SBIN.NS.csv",parse_dates=['Date'])
MRF=pd.read_csv("MRF.NS (1).csv",parse_dates=['Date'])
REL=pd.read_csv("RELIANCE.NS.csv",parse_dates=['Date'])
portfolio_list=[BAJ,REL,SBI,TCS,MRF]
portfolio_dict={'BAJ':BAJ,'REL':REL,'SBI':SBI,'TCS':TCS,'MRF':MRF}
#RETURN ANALYSIS
print(BAJ.index.min(),BAJ.index.max())
#BAJ.sort_index()
#creating a line plot showing the Adj close price over last 1 month
fig,ax=plt.subplots(dpi=150,figsize=(10,3))
print(BAJ.info())
print(REL.info())
print(SBI.info())
print(TCS.info())
print(MRF.info())
BAJ['Adj Close'].plot(ax=ax,label="BAJ")
REL['Adj Close'].plot(ax=ax,label="REL")
SBI['Adj Close'].plot(ax=ax,label="SBI")
TCS['Adj Close'].plot(ax=ax,label="TCS")
plt.legend()
#Creating a function that takes adj close price series
def perc_calc(counter,start_date,end_date):
    if start_date not in counter['Adj Close'].index:
        return f"Start Date not in index"
    if end_date not in counter['Adj Close'].index:
        return f"End Date not in index"
    adj_close_start=counter['Adj Close'][start_date]

    adj_close_end=counter['Adj Close'][end_date]
    change=100*(adj_close_end-adj_close_start)/adj_close_start
    return f"Percent Change: {np.round(change,2)}"
#creating a histogram of the daily returns for each stock in the portfolio
plt.figure(dpi=100,figsize=(10,4))
for stock_name,stock_df in portfolio_dict.items():
    stock_df['Adj Close'].pct_change(1).hist(label=stock_name,alpha=0.3,bins=70)
plt.legend()   
# if a person has invested 1000 bucks in BAJ at the start of time series,you would have about 760 bucks at the end of the time period.
#create a plot that shows the value of 1000 at the start of time series and what value it would have in dollars throughout the rest of the time period
def Returns(stock_name,value):
    ret_BAJ=BAJ['Adj Close'].pct_change(1)[1:]
    ret_SBI=SBI['Adj Close'].pct_change(1)[1:]
    ret_REL=REL['Adj Close'].pct_change(1)[1:]
    ret_TCS=TCS['Adj Close'].pct_change(1)[1:]
    ret_MRF=MRF['Adj Close'].pct_change(1)[1:]
    if stock_name=="BAJ":
       print(ret_BAJ)
       ret=ret_BAJ
    elif stock_name=="SBI":
        print(ret_SBI)
        ret=ret_SBI
    elif stock_name=="REL":
        print(ret_REL)
        ret=ret_REL
    elif stock_name=="TCS":
        print(ret_TCS)
        ret=ret_TCS
    elif stock_name=="MRF":
        print(ret_MRF)
        ret=ret_MRF
        
    cummulative_ret=(ret+1).cumprod()
    cummulative_ret*=value
    print(cummulative_ret)
    cummulative_ret.plot(label=stock_name+"_return")
    plt.legend()
value=int(input("Enter an amount in Rs/-"))

plt.xlim("2020-01-01","2024-01-01")
#Volume Analysis
#creating a plot showing the daily volume of stock traded over the time period of 1 month
#volume analysis for BAJ 
'''BAJ['Total_volume']=BAJ['Adj Close']*BAJ['Volume']
print(BAJ.info())'''
for stock_name,stock_df in portfolio_dict.items():
    stock_df['Total_volume']=stock_df['Adj Close']*stock_df['Volume']
    stock_df['Total_volume'].plot(label=stock_name)
plt.legend()  

#Technical analysis
#Moving average
plt.figure(figsize=(10,3),dpi=150)
def mov_avg(stock_name):
    if stock_name=="BAJ":
        plt.figure(figsize=(10,4),dpi=240)
        BAJ['Adj Close'].rolling(window=30).mean().plot(label='30 Day MA')
        BAJ['Adj Close'].plot(label='Adj Close')
        plt.legend()
    elif stock_name=="REL":
        plt.figure(figsize=(10,4),dpi=240)
        REL['Adj Close'].rolling(window=30).mean().plot(label='30 Day MA')
        REL['Adj Close'].plot(label='Adj Close')
        plt.legend()
    elif stock_name=="SBI":
        plt.figure(figsize=(10,4),dpi=240)
        SBI['Adj Close'].rolling(window=30).mean().plot(label='30 Day MA')
        SBI['Adj Close'].plot(label='Adj Close')
    elif stock_name=="TCS":
        plt.figure(figsize=(10,4),dpi=240)
        TCS['Adj Close'].rolling(window=30).mean().plot(label='30 Day MA')
        TCS['Adj Close'].plot(label='Adj Close')
    elif stock_name=="MRF":
        plt.figure(figsize=(10,4),dpi=240)
        MRF['Adj Close'].rolling(window=30).mean().plot(label='30 Day MA')
        MRF['Adj Close'].plot(label='Adj Close')
mov_avg(name_stock)
#Bollinger Bands 20 days, calculate N-period MA +or- 2 days
fig,ax=plt.subplots(figsize=(10,3),dpi=150)
BAJ['MA']=BAJ['Adj Close'].rolling(20).mean()
BAJ['STD']=BAJ['Adj Close'].rolling(20).std()
BAJ['BOL_LOWER']=BAJ['MA']-2*BAJ['STD']
BAJ['BOL_UPPER']=BAJ['MA']+2*BAJ['STD']
BAJ[['Adj Close','BOL_LOWER','BOL_UPPER']].plot(ax=ax)

#chaikin money flow
fig,ax=plt.subplots(figsize=(10,3),dpi=150)
BAJ['Chaikin_Multiplier']=((BAJ['Close']-BAJ['Low'])-(BAJ['High']-BAJ['Close']))/(BAJ['High']-BAJ['Low'])
BAJ['Moneyflow_volume']=BAJ['Chaikin_Multiplier']*BAJ['Volume']
#accumulation distribution line
BAJ["CMF"]=BAJ['Moneyflow_volume']/BAJ['Volume']
BAJ['CMF'].plot(ax=ax)
#using plotly
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
#get the numbers of rows and columns in the data set
print(df.shape)

#Data Visualization
import seaborn as sns
df.corr()
plt.figure(figsize=(25,15))
sns.heatmap(df.corr(method="pearson"),annot=True)
df.plot(x="Date",y=["Low","High"],color=['r','b'],figsize=[25,15])
plt.fill_between(df['Date'],df['Low'],df['High'],color='k',alpha=0.5)
plt.show()
#visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Adj Close Price History')
plt.plot(df['Adj Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel("Adj close price Rs/-", fontsize=18)
plt.show()
#Training dataset and Testing datasets
#creating a new dataframe with only the Adj close column
data=df.filter(['Adj Close'])
#converting dataframe into numpy array
dataset=data.values
#getting the no of rows to train the model
training_data_len=math.ceil(0.8*len(dataset))
print(training_data_len)
#scaling the dataset
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
print(scaled_data)
#creating the scaled training dataset
train_data=scaled_data[0:training_data_len,:]
#split the training dataset into x_train and y_train
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
#convert the x_train and y_train into numpy array
x_train,y_train=np.array(x_train),np.array(y_train)
print(x_train.shape)
#Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Build the lstm model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#compile the model
model.compile(optimizer="adam",loss="mean_squared_error")
#train the model
model.fit(x_train,y_train,batch_size=1,epochs=8)
#create testing dataset
#create new array containing scaled values
test_data=scaled_data[training_data_len-60:,:]
#creating x_test and y_Test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
#convert the data to a numpy array
x_test=np.array(x_test)
#reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#get the models predicted price values
predictions_test=model.predict(x_test)
predictions_train=model.predict(x_train)
#compute residuals from LSTM model
train_residuals=y_train-predictions_train.ravel()
test_residuals=y_test-predictions_test.ravel()

#Train gradient boosting model on residuals
from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=1)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
gb_model.fit(x_train,train_residuals)
#evaluating the model
gb_train_predictions=gb_model.predict(x_train)
gb_test_predictions=gb_model.predict(x_test)
# Add LSTM and Gradient Boosting predictions to get final predictions
train_predictions = predictions_train.ravel() + gb_train_predictions
test_predictions = predictions_test.ravel() + gb_test_predictions
print(x_test.shape)
print(gb_test_predictions.shape)
#inversing the scaler
predictions_test=scaler.inverse_transform(test_predictions.reshape(-1,1))
predictions_train=scaler.inverse_transform(train_predictions.reshape(-1,1))

#Evaluating the model
#evaluate the model we can do by RMSE
rmse_test=np.sqrt(np.mean(predictions_test-y_test)**2)
print(rmse_test)
#plot the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Prediction']=predictions_test
#visulaize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj close',fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close','Prediction']])
plt.legend(['Train','val','Predicted'],loc='lower right')
plt.show()
print(valid)
#return analysis part
Returns(name_stock,value)
#create a new dataframe
new_df=df.filter(['Adj Close'])
#get the last 30 days of Adj Close price value and convert the df into array
last_30_days=new_df[-30:].values
#scaling the data b/n 0 and 1
last_30_days_scaled=scaler.transform(last_30_days)
#create an empty list
X_test=[]
#append past 30 days in X_test
X_test.append(last_30_days_scaled)
#converting into numpy array
X_test=np.array(X_test)
#reshape the array into 3D
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#get the predicted scaled price
pred_price=model.predict(X_test)
#reversing the scaling
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)
adj_close=df['Adj Close'].iloc[-1]
diff_amt=adj_close-pred_price[0][0]
if diff_amt>0:
     loss=diff_amt*value
     print(f"loss, no need to invest")
else:
    profit=abs(diff_amt*value)
    print(f"profitted amt :{profit}")

#Building an predictive function
def predicted_price(df):
    count=0
    new_df=df.filter(['Adj Close'])
    while count!=30:
        count+=1
        last_30_days=new_df[-30:].values
        last_30_days_scaled=scaler.transform(last_30_days)
        X_test=[]
        X_test.append(last_30_days_scaled)
        X_test=np.array(X_test)
        X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
        pred_price=model.predict(X_test)
        pred_price=scaler.inverse_transform(pred_price)
        pred_list=[]
        pred_list.append(pred_price[0][0])
        ndf=pd.DataFrame(data=pred_list)
        ndf.to_csv(df['Adj Close'],mode='a',header=False)
        print(pred_price)
#predicted_price(df)

