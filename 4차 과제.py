#!/usr/bin/env python
# coding: utf-8

# # 과제4 : 파이썬 기초(4)
# ## 예외처리/내장함수/외장함수
# ### 제출기한 : 9월 30일 월요일 18:59까지

# ### 1. 예외처리

# 다음 빈칸을 채워서 소스 코드를 완성하세요. 
# 
# maria.txt 파일이 있으면 파일의 내용을 읽어서 출력하고, 파일이 없으면 '파일이 없습니다.'를 출력하도록 만드세요. 파일이 없을 때 발생하는 예외는 FileNotFoundError입니다.

# ①  ___________   
# 
#     file = open('maria.txt', 'r')
#     
# ②  __________________________   
# 
#     print('파일이 없습니다.')
#     
# ③  ______   
# 
#     s = file.read()
#     
#     file.close()

# In[2]:


try:
    file = open('maria.txt','r')
except FileNotFoundError:
    print("파일이 없습니다.")
else:
    s = file.read()
    file.close()


# ### 2. 에러발생시켜보기

# allowed = ['가위','바위','보'] 리스트에 없는 값을 rsp 함수에 입력했을 때 에러를 발생시키고 적당한 문구를 표시해라

# In[5]:


def rsp(mine,your):
    allowed = ['가위','바위','보']
    if mine not in allowed:
        raise ValueError
        print("가위,바위,보 중 하나를 내주세요.")
    if your not in allowed:
        raise ValueError
        print("상대방이 아무것도 내지 않았습니다.")
    if mine== '가위' and your == '바위':
        print("상대방이 이겼습니다.")
    if mine== '바위' and your == '보':
        print("상대방이 이겼습니다.") 
    if mine== '보' and your == '가위':
        print("상대방이 이겼습니다.")
    if mine== '가위' and your == '보':
        print("당신이 이겼습니다.") 
    if mine== '바위' and your == '가위':
        print("당신이 이겼습니다.") 
    if mine== '보' and your == '바위':
        print("당신이 이겼습니다.")
    if mine== '가위' and your == '가위':
        print("비겼습니다.")
    if mine== '바위' and your == '바위':
        print("비겼습니다.")
    if mine== '보' and your == '보':
        print("비겼습니다.")        
#함수를 완성시켜 보세요


# In[6]:


try:
    rsp('가위','바')
except ValueError:
    print("잘못된 값을 입력했습니다.")
#예외처리를 해보세요


# ### 4. enumerate와 for문을 이용해 다음 리스트의 홀수 번째 요소만 출력하라.

# In[11]:


ice_cream = ['빠삐코','더위사냥','비비빅','죠스바','누가바']


# In[17]:


for i,name in enumerate(list(ice_cream)):
    if (i+1)%2 == 1:
        print (name)


# ### 5. filter와 lambda를 사용하여 다음 리스트의 요솟값 중 100이하의 수만 가지는 리스트를 만들어라.

# In[18]:


my_list = [93,20,1004,94,104,12]


# In[19]:


list(filter(lambda x: x<100, my_list))


# ### 6. map과 lambda를 사용하여 [-23,13,-199,-29] 리스트의 각 요솟값의 절댓값을 가지는 리스트를 만들어라.

# In[21]:


list(map(lambda x: abs(x), [-23,13,-199,-29]))


# ### 7. 다음과 같은 리스트가 있다. 이 리스트의 개수, 평균, 표준편차, 편차의 평균, 최솟값, 최댓값, 중간값을 구해라.

# In[28]:


my_data=[29,102,38,132,235,23,7,41,31,52,29]


# In[40]:


len(list(my_data))
import numpy as np
np.mean(list(my_data))
np.std(list(my_data))
np.std(list(my_data))
min(list(my_data))
max(list(my_data))
np.median(list(my_data))


# ### 8. 현재 디렉터리 위치를 출력한 후, 바탕화면으로 디렉터리 위치를 변경하라.

# In[56]:


import os
os.getcwd()
os.chdir(C:\Users\min98\OneDrive)


# ### 9. 바탕화면에 있는 파일 중 확장자가 csv인 파일을 모두 찾아서 출력하라.

# In[58]:


import glob
glob.glob(".csv")


# ### 10. 다음과 같은 리스트가 있다. 이 리스트 중 랜덤으로 30개를 test_set에, 나머지 70개를 train_set에 리스트 형태로 넣어라.

# In[63]:


my_set = list(range(100))
print(my_set)


# In[65]:


test_set =[]
train_set = []
import random
random.shuffle(my_set)
for i,name in enumerate(my_set):
    if i<30:
        test_set.append(name)
    else:
        train_set.append(name)
        
print(list(test_set))
print(list(train_set))

