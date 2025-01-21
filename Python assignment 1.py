#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1
print("Vijay Manohar")
print("ST1001")
print("vijaymanohar@gmail.com")


# In[2]:


#2
print("Vijay Manohar\nST1001\nvijaymanohar@gmail.com")


# In[4]:


#3
num1 = 14
num2 = 7
print(f"{num1} + {num2} = {num1 + num2}")
print(f"{num1} - {num2} = {num1 - num2}")
print(f"{num1} * {num2} = {num1 * num2}")
print(f"{num1} / {num2} = {num1 / num2}")


# In[6]:


# 4
for i in range(1, 6):
    print(i)


# In[7]:


#5
print("\"SDK\" stands for \"Software Development Kit\", whereas \"IDE\" stands for \"Integrated Development Environment\".")


# In[8]:


#6
print("python is an \"awesome\" language.")
print("python\n\t2023")
print('I\'m from Entri.\b')
print("\65")  
print("\x65")  
print("Entri", "2023", sep="\n")
print("Entri", "2023", sep="\b")
print("Entri", "2023", sep="*", end="\b\b\b\b")


# In[9]:


# 7
num = 23
textnum = "57"
decimal = 98.3
print(f"Type of num: {type(num)}")
print(f"Type of textnum: {type(textnum)}")
print(f"Type of decimal: {type(decimal)}")

sum_of_variables = num + int(textnum) + decimal
print(f"Sum of variables: {sum_of_variables}")
print(f"Type of sum: {type(sum_of_variables)}")


# In[10]:


# 8
days_in_year = 365
hours_in_day = 24
minutes_in_hour = 60

total_minutes_in_year = days_in_year * hours_in_day * minutes_in_hour
print(f"The total number of minutes in a year is {total_minutes_in_year}.")


# In[11]:


# 9
name = input("Please enter your name: ")
print(f"Hi {name}, welcome to Python programming :)")


# In[12]:


# 10 
pounds = float(input("Please enter amount in pounds: "))
conversion_rate = 1.25  
dollars = pounds * conversion_rate
print(f"£{pounds} are ${dollars:.2f}.")


# In[ ]:




