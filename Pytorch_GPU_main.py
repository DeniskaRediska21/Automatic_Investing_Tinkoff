import time
t=time.perf_counter()
from tinkoff.invest import Client, CandleInterval
import datetime
import numpy as np
import matplotlib.pyplot as plt
import config
import pandas as pd
import numpy as np
from numpy import concatenate

import ta
import pandas

import torch

torch.cuda.empty_cache()

def get_close(from_,to_,token,figi):

    with Client(token) as client:
        r=client.market_data.get_candles(
            figi=figi,
            from_=datetime.datetime(from_.year,from_.month,from_.day),
            to=datetime.datetime(to_.year,to_.month,to_.day),
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
        )
        close=[]
        high = []
        low = []
        volume = []
        open=[]
        for candle in r.candles:
            close.append(candle.close.units+0.000000001*candle.close.nano)
            high.append(candle.high.units+0.000000001*candle.high.nano)
            low.append(candle.low.units+0.000000001*candle.low.nano)
            open.append(candle.open.units+0.000000001*candle.open.nano)
            volume.append(candle.volume)
            
    return np.array(close),np.array(high),np.array(low),np.array(volume),np.array(open)
    
    
def get_several_days(n,token,figi,nn=0):
    t=datetime.datetime.today().replace(microsecond=0,second=0,hour=0,minute=0)
    close=[]
    high = []
    low = []
    volume = []
    open = []
    for i in range(n):
        cl,hi,lo,vo,op = get_close(t-datetime.timedelta(days=nn+1),t-datetime.timedelta(days=nn),token,figi)
        close.append(cl)
        high.append(hi)
        low.append(lo)
        volume.append(vo)
        open.append(op)
        t=t-datetime.timedelta(days=1)
        t=t.replace(microsecond=0,second=0,hour=0,minute=0)
    return close, high, low, volume, open


# ---------------------------------------------------------------------
def chunks(signal,n,nn, nn_, profit_margin, loss_margin, broker = 0.0005):
    signal_split = []
    y = []
    y_ = []
    signal=np.reshape(signal,[-1,max(np.shape(signal))])
    for i in range(max(np.shape(signal))-n- nn):
        signal_split.append(signal[:,i:i+n])
        y.append(int(signal[0,i+n]*(1+profit_margin*broker)<np.max(signal[0,i+n:i+nn+n])))
        y_.append(int(signal[0,i+n]/(1+loss_margin*broker)>np.min(signal[0,i+n:i+nn_+n])))
        
    return np.array(signal_split),np.array(y),np.array(y_)


def get_colomn(lst,n,nn):
    tmp = np.zeros([len(lst),nn])
    for i in range(0,len(lst),1):
        tmp[i] = lst[i][n]
    return tmp

def get_training(signal, close,n_test,n_train,n_test_, profit_margin=3, lose_margin=3):
    input, output_, output2_= chunks(signal,n_train,n_test,n_test_,profit_margin=profit_margin,loss_margin=lose_margin)
    input_, output, output2= chunks(close,n_train,n_test,n_test_,profit_margin=profit_margin,loss_margin=lose_margin)
    input = torch.from_numpy(np.array(input))
    output = torch.from_numpy(np.array(output))
    output2 = torch.from_numpy(np.array(output2))
    input =input.float() 
    output =output.long() 
    output2 =output2.long()
    mean_=[] 
    norm_=[]
    for i in range(input.size()[1]):
        mean_.append(input[:,i,:].mean())
        input[:,i,:]=(input[:,i,:]-mean_[i])
        norm_.append(input[:,i,:].abs().max())
        input[:,i,:]=(input[:,i,:]/norm_[i])
    return input,output,mean_,norm_, output2


from torch import nn as nn 
# def get_model(n_inputs,n_train):
#     model = nn.Sequential(
#             nn.LayerNorm(n_train),
#             nn.Conv1d(n_inputs,256,5),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv1d(256,128,5),
#             nn.ReLU(),
#             nn.Conv1d(128,64,5),
#             nn.ReLU(),
#             nn.Conv1d(64,32,5),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(84,100),
#             nn.ReLU(),
#             nn.MaxPool1d(5),
#             nn.Flatten(),
#             nn.Linear(640,150),
#             nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(150,2),
#             nn.ReLU()
#             )
#     model.to('cuda')
#     return model

def get_model(n_inputs,n_train):
    model = nn.Sequential(
            nn.LayerNorm(n_train),
            nn.Conv1d(n_inputs,150,5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(150,128,5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128,64,5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,32,5),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(5,100),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(1600,500),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(500,150),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(150,2),
            nn.ReLU()
            )
    model.to('cuda')
    return model

def train_model(model, input_train,output_train,Epocs=10,lr=0.7,loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1,1]).float().to('cuda')),val_split=0.3,l2_lambda=0.00008):
    import torch.optim as optim
    I=torch.randperm(input_train.size()[0])
    n=int(np.round(val_split*input_train.size()[0]))
    val_input_train = input_train[I[:n]].to('cuda')
    val_output_train = output_train[I[:n]].to('cuda')
    
    
    input_train = input_train[I[n:]].to('cuda')
    output_train = output_train[I[n:]].to('cuda')
    
    min_train_val_loss=np.inf
    # move the tensor to gpu
    optimizer=optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    model_log=[]
    # train the model for 10 epochs
    for epoch in range(Epocs):
        # forward pass
        train_pred = model(input_train)
        # compute loss

        loss = loss_fn(train_pred, output_train)
        # l2_lambda = l2_lambda
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        val_pred = model(val_input_train)
        val_loss=loss_fn(val_pred, val_output_train)
        if val_loss < min_train_val_loss:
            min_train_val_loss=val_loss
            model_log=model
        # else:
        if val_loss > 1:
            print(f'Прищлось прерваться на эпохе: {epoch}')
            break
        if np.mod(epoch,10)==0:
            print(f'{epoch}:      {loss}      {val_loss}     {min_train_val_loss}')
        #torch.cuda.empty_cache()

    
    
    train_pred=np.argmax(model(val_input_train).detach().cpu().numpy(),1)
    output_train=val_output_train.detach().cpu().numpy()
    err=output_train.squeeze()-train_pred.squeeze()
    acc=1-np.mean(np.abs(err))
    err=np.sum(err[err<0])
    print(f'Минимальные потери: {min_train_val_loss}')
    print(f'Accuracy = {acc*100} %')
    print(f'Всего удачных возможностей на обучении: {np.sum(output_train)}')
    print(f'Распознанные возможности на обучающей выборке: {np.sum(train_pred)}')
    print(f'Было совершено ошибочно: {np.abs(err)}\n')
    input_train.detach()
    return model_log



def get_prediction(model,input,mean_=0,norm_=1):
    input=np.reshape(input,[-1,np.min(input.shape)])
    input = torch.from_numpy(np.array(input))
    try:
        nn=input.shape[1]
    except: 
        nn=1
        
    input=input.reshape(1,nn,input.shape[0])
    input =input.float() 
    for i in range(input.shape[1]):
        input[:,i,:]=(input[:,i,:]-mean_[i])
        input[:,i,:]=(input[:,i,:]/norm_[i])
    return np.argmax(model(torch.from_numpy(np.array(input)).float().to('cuda')).detach().cpu().numpy())

import random



token =config.token
figi='BBG006L8G4H1' # YNDX
# figi = 'BBG00178PGX3' #VKCO
#figi='BBG000000001' # TINKOFF
#figi='BBG00Y91R9T3' # OZON
id='2059195636'
n_days=10
lag=1
close, high, low, volume, open=get_several_days(n_days,token,figi,lag)
close_concat=np.concatenate(close)
diff_=(np.squeeze(list(np.diff([close_concat[0],close_concat]))))
high_concat=np.concatenate(high)
low_concat=np.concatenate(low)
volume_concat=np.concatenate(volume)
open_concat=np.concatenate(open)
print(len(close_concat))
srsi=np.array(ta.momentum.stochrsi(pd.Series(close_concat),fillna =True))
kama_=np.array(ta.momentum.KAMAIndicator(pd.Series(close_concat),fillna =True).kama())
ua=np.array(ta.momentum.UltimateOscillator(pd.Series(high_concat),pd.Series(low_concat),pd.Series(close_concat),fillna =True).ultimate_oscillator())
tsi_=np.array(ta.momentum.TSIIndicator(pd.Series(close_concat),fillna =True).tsi())
macd_=np.array(ta.trend.macd(pd.Series(close_concat),fillna =True))



from sklearn.decomposition import PCA


signal=np.array([close_concat,low_concat, high_concat, volume_concat,srsi,kama_,tsi_,ua,macd_])


pca = PCA(n_components=3)
pca.fit(np.transpose(signal))
signal_transformed=np.transpose(pca.transform(np.transpose(signal)))
signal=signal_transformed




broker=0.0005
n_train = 100
n_test=20
n_test_ = 10
profit_margin=5
input_train,output_train,m,n_,output_train2=get_training(signal,close_concat,n_test,n_train,n_test_,profit_margin=profit_margin)
model=get_model(input_train.size()[1],n_train)
model=train_model(model,input_train,output_train,Epocs=5000,lr=0.1,l2_lambda=0.0000)

# _____________________________________________________________________________________________________
def get_last_close(token,figi,n=1):
    with Client(token) as client:
        to_=datetime.datetime.now()
        to_.replace(second=0, microsecond=0)
        
        from_=to_-datetime.timedelta(minutes=n,seconds=5)
        from_.replace(second=0, microsecond=0)
        
        r=client.market_data.get_candles(
            figi=figi,
            from_=from_,
            to=to_,
            # from_=datetime.datetime(from_.year,from_.month,from_.day,from_.hour,from_.minute),
            # to=datetime.datetime(to_.year,to_.month,to_.day,to_.hour,to_.minute),
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN
        )
        close=[]
        high=[]
        low=[]
        volume=[]
        for candle in r.candles:
            close.append(candle.close.units+0.000000001*candle.close.nano)
            high.append(candle.high.units+0.000000001*candle.high.nano)
            low.append(candle.low.units+0.000000001*candle.low.nano)
            volume.append(candle.volume)
            
    return np.array(close),np.array(high),np.array(low), np.array(volume)


now=datetime.datetime.now()
if now>datetime.datetime(year=now.year,month=now.month,day=now.day,hour=9,minute=50)+datetime.timedelta(minutes=101):
    print('Using history')
    close, high, low, volume=get_last_close(token,figi,n=120)
else:
    print('Collecting history...')
    close=[]
    high=[]
    low=[]
    volume=[]

close=np.array(close) 
y_log=[]


quantity=1

import time
t=time.perf_counter()-60
buys = np.empty(np.shape(close))
buys[:] = np.nan
sels = np.empty(np.shape(close))
sels[:] = np.nan
buy_price=np.inf
b_flag=0
prev=0
prev_=0
input_test_prev=np.inf
from tinkoff.invest import Client, RequestError, OrderDirection, OrderType, Quotation
try:
    while True:
        if time.perf_counter()-t>60:
            t=time.perf_counter()
            last_close, last_high, last_low, last_volume=get_last_close(token,figi)
            if np.size(last_close)!=0:
                last_close=last_close[-1]
                
            if np.size(last_volume)!=0:
                last_volume=last_volume[-1]
                
            if np.size(last_low)!=0:
                last_low=last_low[-1]
                
            if np.size(last_high)!=0:
                last_high=last_high[-1]
                
            close=np.append(close,last_close)
            high=np.append(high,last_high)
            low=np.append(low,last_low)
            volume=np.append(volume,last_volume)
            if len(close)>n_train:
                srsi=np.array(ta.momentum.stochrsi(pd.Series(close),fillna =True))
                kama_=np.array(ta.momentum.KAMAIndicator(pd.Series(close),fillna =True).kama())
                tsi_=np.array(ta.momentum.TSIIndicator(pd.Series(close),fillna =True).tsi())
                ua=np.array(ta.momentum.UltimateOscillator(pd.Series(high),pd.Series(low),pd.Series(close),fillna =True).ultimate_oscillator())
                diff_=np.diff(np.concatenate([[close[0]],close]))
                macd_=np.array(ta.trend.macd(pd.Series(close),fillna =True))
                signal=np.array([diff_,low,high,volume, srsi,kama_,tsi_,ua,macd_])
                signal_transformed=np.transpose(pca.transform(np.transpose(signal)))
                signal=signal_transformed
                
                y=get_prediction(model,signal[:,np.shape(signal)[1]-n_train:],m,n_)
                signal_=np.array([-(signal_ - m_.numpy())+m_.numpy() for signal_, m_ in zip(signal[:,np.shape(signal)[1]-n_train:], m)])
                y2=get_prediction(model,signal_,m,n_)
                y_log=np.append(y_log,y)                    
                
                
                    
                if  b_flag==0 and (np.size(last_close)!=0) and (((input_test_prev<=close[-1]) and (y==1)and (kama_[-1]>close[-1]))):
                    try:
                        buys=np.append(buys,1)
                        # 
                    
                        with Client(token) as client:
                            r = client.orders.post_order(
                                order_id=str(datetime.datetime.now().timestamp()),
                                figi=figi,
                                quantity=quantity,
                                account_id=id,
                                direction=1,
                                order_type=OrderType.ORDER_TYPE_MARKET
                            )
                        b_flag=1
                        buy_price=r.executed_order_price.units+0.000000001*r.executed_order_price.nano
                    except:
                        buys=np.append(buys,np.nan)
                else:
                    buys=np.append(buys,np.nan)
                    
                if  b_flag>0:
                    b_flag+=1
                prev=y
                
                if (np.size(last_close)!=0) and (((b_flag>n_test) and (y2==1) and (last_close>buy_price*(1+broker*2.5)))  or (b_flag!=0 and (y2==1) and (last_close>buy_price*(1+broker*4)))):
                    sels=np.append(sels,1)
                    with Client(token) as client:
                        r = client.orders.post_order(
                        order_id=str(datetime.datetime.now().timestamp()),
                        figi=figi,
                        quantity=quantity,
                        account_id=id,
                        direction=2,
                        order_type=OrderType.ORDER_TYPE_MARKET
                    )
                    b_flag=0
                else:
                    sels=np.append(sels,np.nan)
                input_test_prev=close[-1]
                prev_=y2
                print(f'y:{y}, y_:{y2}   ,   close: {close[-1]}   ,   kama: {kama_[-1]}  ,   buy: {buys[-1]}    ,   sell: {sels[-1]}, {b_flag}')
            else:
                print(f'Придется подождать: {n_train-len(close)}')
except KeyboardInterrupt:
    pass
