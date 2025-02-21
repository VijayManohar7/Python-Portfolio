#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1
class Course:
    def __init__(self, course_code, course_name, credit_hours):
        self.course_code = course_code
        self.course_name = course_name
        self.credit_hours = credit_hours

    def display_info(self):
        return f"Course Code: {self.course_code}, Name: {self.course_name}, Credit Hours: {self.credit_hours}"

class CoreCourse(Course):
    def __init__(self, course_code, course_name, credit_hours, required_for_major):
        super().__init__(course_code, course_name, credit_hours)
        self.required_for_major = required_for_major

    def display_info(self):
        return super().display_info() + f", Required for Major: {'Yes' if self.required_for_major else 'No'}"

class ElectiveCourse(Course):
    def __init__(self, course_code, course_name, credit_hours, elective_type):
        super().__init__(course_code, course_name, credit_hours)
        self.elective_type = elective_type

    def display_info(self):
        return super().display_info() + f", Elective Type: {self.elective_type}"

core_course = CoreCourse("CS101", "Introduction to Computer Science", 3, True)
elective_course = ElectiveCourse("HIST200", "World History", 3, "liberal arts")

print(core_course.display_info())
print(elective_course.display_info())


# In[8]:


# employee.py

class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

    def get_name(self):
        return self.name

    def get_salary(self):
        return self.salary


# In[9]:


# main.py

from employee import Employee

# Creating an Employee object
emp = Employee("John Doe", 50000)

# Displaying Employee details
print("Employee Name:", emp.get_name())
print("Employee Salary:", emp.get_salary())


# In[ ]:




