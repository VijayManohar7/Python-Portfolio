#!/usr/bin/env python
# coding: utf-8

# In[7]:


# 1
try:
    with open('Assignment_5.txt', 'w') as file:
        file.write('Hello all')
    print("File created and written successfully.")
except Exception as e:
    print("An error occurred:", e)


# In[8]:



with open('Assignment_5.txt', 'r') as abc:
    content=abc.read()
    print("content of the file: ")
    print(content)


# In[11]:


#2
def create_sample_file(filename):
    with open(filename, 'w') as file:
        file.write("Hello all! This is a sample file for testing.\n")
        file.write("Let's test file handling and exception handling.\n")
    print(f"{filename} has been created with sample content.")

def copy_file(source, destination):
    try:
        with open(source, 'r') as src, open(destination, 'w') as dest:
            dest.write(src.read())
        print(f"File copied successfully from {source} to {destination}.")
    except FileNotFoundError:
        print(f"Error: Source file '{source}' not found.")

def verify_file(filename):
    try:
        with open(filename, 'r') as file:
            print(f"\nContents of {filename}:")
            print(file.read())
    except FileNotFoundError:
        print(f"Error: {filename} not found.")

source_file = "Assignment_5.txt"
destination_file = "Copied_Assignment_5.txt"


# In[12]:


#3
def count_words(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            words = content.split()  
            word_count = len(words)
            print(f"Total number of words in '{filename}': {word_count}")
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")

filename = "Assignment_5.txt"  
count_words(filename)


# In[13]:


#4
def convert_to_integer():
    try:
        user_input = input("Enter a number: ")  
        number = int(user_input)  
        print(f"Successfully converted to integer: {number}")
    except ValueError:
        print("Error: Invalid input! Please enter a valid integer.")

convert_to_integer()


# In[14]:


#5
def check_positive_numbers():
    try:
        user_input = input("Enter a list of integers separated by spaces: ")
        numbers = list(map(int, user_input.split()))  
        
        for num in numbers:
            if num < 0:
                raise ValueError("Negative numbers are not allowed!")

        print("All numbers are positive:", numbers)
    
    except ValueError as e:
        print(f"Error: {e}")

check_positive_numbers()


# In[15]:


#6
def compute_average():
    try:
        user_input = input("Enter a list of integers separated by spaces: ")
        numbers = list(map(int, user_input.split()))  

        if not numbers:  
            raise ValueError("The list cannot be empty.")

        average = sum(numbers) / len(numbers)  
        print(f"The average of the numbers is: {average:.2f}")

    except ValueError as e:
        print(f"Error: {e}") 

    finally:
        print("Program execution finished.")  

compute_average()


# In[16]:


#7
def write_to_file():
    try:
        filename = input("Enter the filename: ")  
        text = input("Enter the text to write into the file: ")  

        with open(filename, 'w') as file:  
            file.write(text)

        print("Welcome! The file has been successfully created and written.")

    except Exception as e:  
        print(f"An error occurred: {e}")

write_to_file()


# In[ ]:




