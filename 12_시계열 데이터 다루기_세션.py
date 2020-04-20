#!/usr/bin/env python
# coding: utf-8

# ## datetime 오브젝트 사용하기

# In[ ]:


from datetime import datetime


# In[ ]:


now1 = datetime.now() 
print(now1)


# In[ ]:


now2 = datetime.today()
print(now2) 


# In[ ]:


t1 = datetime.now() 
t2 = datetime(1970, 1, 1)
t3 = datetime(1970, 12, 12, 13, 24, 34)

print(t1)
print(t2)
print(t3)


# In[ ]:


diff1 = t1 - t2

print(diff1)
print(type(diff1))


# In[ ]:


diff2 = t2 - t1

print(diff2)
print(type(diff2))


# 
# ## 문자열을 datetime 오브젝트로 변환하기

# In[11]:


import pandas as pd 
import os
ebola = pd.read_csv('C:/Users/min98/OneDrive/바탕 화면/ESAA/0420수업자료2/country_timeseries.csv')


# In[4]:


print(ebola.info())


# In[ ]:


ebola['Date'] = pd.to_datetime(ebola['Date'])
print(ebola.info())


# In[ ]:


test_df1 = pd.DataFrame({'order_day':['01/01/15', '02/01/15', '03/01/15']})

test_df1['date_dt1'] = pd.to_datetime(test_df1['order_day'], format='%d/%m/%y')
test_df1['date_dt2'] = pd.to_datetime(test_df1['order_day'], format='%m/%d/%y')
test_df1['date_dt3'] = pd.to_datetime(test_df1['order_day'], format='%y/%m/%d')

print(test_df1)


# In[ ]:


test_df2 = pd.DataFrame({'order_day':['01-01-15', '02-01-15', '03-01-15']})
test_df2['date_dt'] = pd.to_datetime(test_df2['order_day'], format='%d-%m-%y')

print(test_df2)


# 
# ## 시계열 데이터를 구분해서 추출

# In[ ]:


now = datetime.now()
print(now)


# In[ ]:


nowDate = now.strftime('%Y-%m-%d')
print(nowDate)


# In[ ]:


nowTime = now.strftime('%H:%M:%S')
print(nowTime) 


# In[ ]:


nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
print(nowDatetime) 


# ## datetime 오브젝트로 변환하려는 열을 지정하여 데이터 집합 불러오기

# In[ ]:


ebola['Date']=pd.to_datetime(ebola['Date'])


# 
# ## datetime 오브젝트에서 날짜 정보 추출하기

# In[ ]:


date_series = pd.Series(['2018-05-16', '2018-05-17', '2018-05-18'])


# In[ ]:


#list 를 date time 형태로 변환하기.
d1 = pd.to_datetime(data_series)


# In[ ]:


print(d1)


# In[ ]:


print(d1[0].year)


# In[ ]:


print(d1[0].month)


# In[ ]:


print(d1[0].day)


# In[ ]:


#위에서 불러들인 ebola 자료에서 date변수를 활용해 새로운 column year로 만들어주기
ebola['year'] = ebola['Date']apply(lambda x:x.year)


# ## dt 접근자로 간단하게 해결하기

# In[ ]:


print(ebola['Date'][3].year)


# In[ ]:


print(ebola['Date'][3].month)


# In[ ]:


print(ebola['Date'].year) #error


# In[ ]:


ebola['Date'].dt.year


# In[ ]:


#dt 접근자를 활용해 ebola data에 month 만들어 주기
ebola['month']= ebola.Date.dt.month


# In[ ]:


ebola['month2'], ebola['day2'] = (ebola.Date.dt.month,ebola.Date.dt.day)


# In[ ]:


ebola[['month2','day2']].head()


# In[ ]:


print(ebola.info())


# ## 에볼라 최초 발병일 계산하기

# In[ ]:


#ebola data (뒤에서) 5행 5열 가져오기
ebola.iloc[-5:,:5]


# In[ ]:


# 에볼라의 최초 발병일을 구하고 최초 발병일로부터 몇일 차이가 나는지를 outbreak_d column에 계산하여 넣기
ebola['outbreak_d']=ebola.Date - ebola.Date.min()


# In[ ]:


print(ebola['Date'].min())
print(type(ebola['Date'].min()))


# ### 파산한 은행의 개수 계산하기

# In[9]:


banks = pd.read_csv('C:/Users/min98/OneDrive/바탕 화면/ESAA/0420수업자료2/banklist.csv') 
banks.head()


# In[12]:


banks = pd.read_csv('C:/Users/min98/OneDrive/바탕 화면/ESAA/0420수업자료2/banklist.csv', parse_dates=[5, 6]) 
print(banks.info())


# In[14]:


#은행 파산의 분기와, 연도를 계산해 closing_quarter, closing_year column에 저장하기
banks['closing_quarter'],banks['closing_year'] = (banks['Closing Date'].dt.quarter, banks['Closing Date'].dt.year)


# In[15]:


closing_year = banks.groupby(['closing_year']).size()
print(closing_year)


# In[16]:


#각 연도의 분기별로 파산된 은행의 갯수 구하기
closing_year_quarter = banks.groupby(['closing_year', 'closing_quarter']).size()
print(closing_year_quarter)


# In[17]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax = closing_year.plot() 
plt.show()

#fig, ax = plt.subplots() 
ax = closing_year_quarter.plot() 
plt.show()


# 
# ## 테슬라 주식 데이터로 시간 계산하기

# In[ ]:


tesla = pd.read_csv('../data/tesla_stock_quandl.csv', parse_dates=[0])
print(tesla.info())


# In[ ]:


tesla.head()


# In[ ]:


#2010년 6월에 해당하는 데이터만 가져오세요


# 
# ## datetime 오브젝트를 인덱스로 설정하여 데이터 추출하기

# In[ ]:


tesla.index = tesla['Date']
print(tesla.index)


# In[ ]:


print(tesla['2015'].iloc[:5, :5])


# In[ ]:


#index 를 활용하여 위와 같이 2010 6월 데이터를 앞에서부터 5번째 column까지 가져오기


# 
# ## 시간 간격을 인덱스로 설정하여 데이터 추출하기

# In[ ]:


tesla['ref_date'] = tesla['Date'] - tesla['Date'].min()
tesla.head()


# In[ ]:


tesla.index = tesla['ref_date']
tesla.iloc[:5, :5]


# In[ ]:


tesla['5 days':].iloc[:5, :5]


# 
# ## 시간 범위 생성하여 인덱스로 지정하기

# In[ ]:


ebola = pd.read_csv('../data/country_timeseries.csv', parse_dates=[0]) 
ebola.iloc[:5, :5]


# In[ ]:


ebola.iloc[-5:, :5]


# In[ ]:


head_range = pd.date_range(start='2014-12-31', end='2015-01-05') 
print(head_range)


# In[ ]:


ebola_5 = ebola.head()
ebola_5.index = ebola_5['Date']
ebola_5.reindex(head_range)

print(ebola_5.iloc[:5, :5])


# 
# ## 시간 범위의 주기 설정하기

# In[ ]:


print(pd.date_range('2017-01-01', '2017-01-07', freq='B'))


# 
# ## 에볼라의 확산 속도 비교하기

# In[ ]:


import matplotlib.pyplot as plt

ebola.index = ebola['Date']

fig, ax = plt.subplots() #그래프 크기 키우기
ax = ebola.iloc[0:, 1:].plot(ax=ax)
ax.legend(fontsize=7, loc=2, borderaxespad=0.) 
plt.show()


# In[ ]:


ebola_sub = ebola[['Day', 'Cases_Guinea', 'Cases_Liberia']] 
print(ebola_sub.tail(10))


# ### 3. 그래프를 그리기 위한 데이터프레임 준비하기

# In[ ]:


ebola = pd.read_csv('../data/country_timeseries.csv',  parse_dates=['Date'])
print(ebola.head().iloc[:, :5])


# In[ ]:


#Date index 만들어 주기
ebola.index = ebola.Date


# In[ ]:


new_idx = pd.date_range(ebola.index.min(), ebola.index.max())
new_idx


# In[ ]:


new_idx = reversed(new_idx)
print(new_idx)


# In[ ]:


ebola = ebola.reindex(new_idx)


# In[ ]:


ebola.head()


# In[ ]:


print(ebola.tail().iloc[:, :5])


# ### 7. 각 나라의 에볼라 발병일을 같은 시점으로 옮겨 비교하기

# In[ ]:


#각 column별로 마지막 valid index 구하기
last_valid = ebola.apply(pd.Series.last_valid_index)


# In[ ]:


#각 나라별로 첫 valid index date 구하기
first_valid = ebola.apply(pd.Series.first_valid_index)


# In[ ]:


last_valid


# In[ ]:


earliest_date = ebola.index.min() 
print(earliest_date)


# In[ ]:


#전세계적 최초 발병 날짜로부터 나라별 최초 발병 날짜까지의 일수 구하기
shift_values = last_valid - earliest_date
print(shift_values)


# In[ ]:


#각 나라의 최초 발병일을 동일하게 한 dataframe다시 만들기
ebola_dict = {}
for idx.col in enumerate(ebola):
    d = shift_values[idx].days
    shifted = ebola[col].shift(d)
    ebola_dict[col] = shifted


# In[ ]:


ebola_shift = pd.DataFrame(ebola_dict)


# In[ ]:


ebola_shift.tail()


# In[ ]:


ebola_shift.index = ebola_shift['Day'] 
ebola_shift = ebola_shift.drop(['Date', 'Day'], axis=1)

ebola_shift.tail()


# In[ ]:


fig, ax = plt.subplots(figsize=(8,6)) 
ax = ebola_shift.iloc[:, :].plot(ax=ax)
ax.legend(fontsize=7, loc=2, borderaxespad=0.) 

#plt.ylim(0,2000)
plt.show()


# In[ ]:




