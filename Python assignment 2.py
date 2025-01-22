#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Q1
random_numbers = [10, 23, 45, 67, 89]
print("Original list:", random_numbers)

# Q2
random_numbers.extend([15, 33, 55])
print("Updated list:", random_numbers)

# Q3 
for number in random_numbers:
    print(number)


# In[2]:


# Q1
person = {'name': 'John', 'age': 25, 'address': 'New York'}
print("Original dictionary:", person)

# Q2
person['phone'] = '1234567890'
print("Updated dictionary:", person)


# In[3]:


# Q1
numbers_set = {1, 2, 3, 4, 5}
print("Original set:", numbers_set)

# Q2
numbers_set.add(6)
print("Set after adding 6:", numbers_set)

# Q3
numbers_set.discard(3)  
print("Set after removing 3:", numbers_set)


# In[4]:


# Q1
numbers_tuple = (1, 2, 3, 4)
print("Tuple:", numbers_tuple)

# Q2
print("Length of the tuple:", len(numbers_tuple))


# In[ ]:




