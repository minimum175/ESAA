#!/usr/bin/env python
# coding: utf-8

# ### 1-1. 비어있는 사람(Human)클래스를 "정의" 하라

# In[14]:


class Human:
    pass


# ### 1-2. 사람(Human)클래스를 "생성"하고 areum변수에 할당해라

# In[15]:


areum = Human()


# ### 1-3. 사람(Human)클래스에 "응애응애"를 출력하는 생성자를 추가해라

# In[ ]:


class Human:
    def __init__(self):
        print("응애응애")


# ### 1-4. 사람(Human)클래스에 (이름,나이,성별)을 받는 생성자를 추가하여라

# In[26]:


class Human:
    def __init__(self,name,age,sex):
        self.name = name
        self.age = age
        self.sex = sex


# ### 1-5. 위에서 생성한 인스턴스의 이름, 나이, 성별을 출력하여라. 인스턴스 변수에 접근하여 값을 얻어 오세요.

# In[28]:


areum = Human("송민경", 23, "여성")
print(areum.name, areum.age, areum.sex)


# ### 1-6.사람 (Human) 클래스에서 이름, 나이, 성별을 출력하는 who() 메소드를 구현하여라.

# In[29]:


class Human:
    def __init__(self, name, age, sex):
        self.name = name
        self.age = age
        self.sex = sex
    def who(self):
        print(self.name, self, age, self.sex)


# ### 1-7.사람 (Human) 클래스에 (이름, 나이, 성별)을 받는 setInfo 메소드를 추가하여라.

# In[32]:


class Human: 
    def setInfo(self, name, age, sex): 
        self.name=name 
        self.age=age 
        self.sex=sex 

areum2 = Human() 
areum2.setInfo("송민경", 23, "여자") 
print(areum2.name, areum2.age, areum2.sex)


# ### 2.다음은 Calculator 클래스이다.

# In[34]:


class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val


# #### 위 클래스를 상속하는 UpgradeCalculator를 만들고 값을 뺄 수 있는 minus 메서드를 추가해 보자. 즉 다음과 같이 동작하는 클래스를 만들어야 한다.

# In[35]:


class UpgradeCalculator(Calculator):
    def minus(self, val):
        self.value -= val


# ### 3. 객체변수 value가 100 이상의 값은 가질 수 없도록 제한하는 MaxLimitCalculator 클래스를 만들어 보자. 즉 다음과 같이 동작해야 한다.

# In[ ]:


#cal = MaxLimitCalculator()
#cal.add(50) # 50 더하기
#cal.add(60) # 60 더하기

#print(cal.value) # 100 출력


# #### 단 반드시 다음과 같은 Calculator 클래스를 상속해서 만들어야 한다.

# In[36]:


class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, val):
        self.value += val
        
class MaxLimitCalculator(Calculator):
    def add(self, val):
        self.value += val
        if self.value >100:
            self.value=100

