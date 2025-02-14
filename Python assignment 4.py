#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1
# The len() function returns the number of items in an object (e.g., list, string, tuple, etc.)
my_list = [1, 2, 3, 4, 5]
print("Length of list:", len(my_list))


# In[3]:


# 2
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")


# In[4]:


# 3
def find_maximum(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

print("Maximum number:", find_maximum([3, 7, 2, 9, 5]))


# In[5]:


# Local vs Global Variables
# A global variable is defined outside any function and can be accessed anywhere in the code.
# A local variable is defined inside a function and is only accessible within that function.
global_var = "I am global"

def demo_function():
    local_var = "I am local"
    print("Inside function:", local_var)
    print("Inside function:", global_var)

demo_function()
print("Outside function:", global_var)


# In[6]:


# 5
def calculate_area(length, width=5):
    return length * width

print("Area with both arguments:", calculate_area(10, 4))
print("Area with default width:", calculate_area(10))


# In[ ]:




