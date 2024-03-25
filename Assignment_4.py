
from dataclasses import dataclass
@dataclass
class Student:
    name: str
    dept: str
    reg_no: int
student_details = Student('ram', 'AI&DS', 465464247356)
print("name:",student_details.name)
print("dept:",student_details.dept)
print("reg_no:",student_details.reg_no)

'''
output:
name: ram
dept: AI&DS
reg_no: 465464247356'''

# * arguments
class Laptop:
    def __init__(self,*element):
        self.type=element[0]
        self.color=element[1]

lenovo=Laptop("personal_laptop","black")
apple=Laptop("office_laptop","grey")
samsang=Laptop("gaming_laptop","white")
print(lenovo.type)
print(samsang.color)
print(apple.type)
'''
output:
personal_laptop
white
office_laptop
'''

# **kwarg
class Laptop:
    def __init__(self,**position):
        self.type=position['a']
        self.color=position['b']
lenovo=Laptop(a="personal_laptop",b="black")
apple=Laptop(a="office_laptop",b="grey")
samsang=Laptop(a="gaming_laptop",b="white")
print(lenovo.color)
print(samsang.color)
print(apple.type)
'''
output:
black
white
office_laptop
'''



# using if condition
def classroom(person):
    if (person=='girl'):
        print("go left")
    elif(person=='boy'):
        print("go rignt")
classroom(person='girl')
#output:go left

#key arguments
def login(email,password,name,phone_no):
    return f"Email: {email} /n Password: {password} /n Name: {name} /n Phone.no{phone_no}"   

page=login('fhgff@gmail.com', '*****', 'ramu', 878785655)
print(page)
'''
output:
('fhgff@gmail.com', '*****', 'ramu', 878785655)
'''

class Register:
    def __init__(self, name, email,phone_no):
        self.name = name
        self.email= email
        self.phone_no= phone_no
    def __str__(self):
        return f"Name:{self.name} \n Email:{self.email} \n Phone_no:{self.phone_no}"
   
web = Register("ramu","fhgff@gmail.com", 878785655)
print(web)
'''
output:
Name:ramu
Email:fhgff@gmail.com
Phone_no:878785655
'''

import math
def find_factorial_value(n):
    fact= math.factorial(n)
    return fact

print("factorial value:" ,find_factorial_value(9))
'''
output:
factorial value: 362880
'''

def dict_operation(my_dict):
    #get()
    my_dict.get("age")
    print("get age:",my_dict)
    #keys()
    my_dict.keys()
    print("keys:",my_dict)
    #values()
    my_dict.values()
    print("values:",my_dict)
    #pop()
    my_dict.pop("hobby")
    print("pop hobby:",my_dict)
my_dict = {"name":"rohan","age":25,"hobby":"reading","reg_no":54653465}
print("current_dict:", my_dict)
dict_operation(my_dict)
'''
output:
current_dict: {'name': 'rohan', 'age': 25, 'hobby': 'reading', 'reg_no': 54653465}
get age: {'name': 'rohan', 'age': 25, 'hobby': 'reading', 'reg_no': 54653465}
keys: {'name': 'rohan', 'age': 25, 'hobby': 'reading', 'reg_no': 54653465}
values: {'name': 'rohan', 'age': 25, 'hobby': 'reading', 'reg_no': 54653465}
pop hobby: {'name': 'rohan', 'age': 25, 'reg_no': 54653465}
'''

#name error
def value(object):
    try:
        value(0)
    except (NameError):
        print("variable name is differ")
a=level(hi)

#output:NameError: name 'level' is not defined.

#zerodivision error
def zeroerror(a):
    try:
        return a/0
    except(ZeroDivisionError):
        print("denominator presents zero ")
zeroerror(10)        
#output:denominator presents zero

#type error
def value2(input_element):
    try:
        return input_element + 12
    except(TypeError):
        print("something went wrong")
value2('hello')
#output: something went worng

#pandas
import pandas as pd

def sales():
    my_dataset={'cars':['bmw','thar','mini cuper'], 'passings':[3,4,6]}
    myvar=pd.DataFrame(my_dataset)
    return myvar
print(sales())

'''
output:
         cars  passings
0         bmw         3
1        thar         4
2  mini cuper         6
'''

def scope_test():
    def local_var():
        spam = "local spam"

    def nonlocal_var():
        nonlocal spam
        spam = "nonlocal spam"

    def global_var():
        global spam
        spam = "global spam"

    spam = "spam"
    local_var()
    print(" local scope:", spam)
    nonlocal_var()
    print(" nonlocal scope:", spam)
    global_var()
    print(" global scope:", spam)

scope_test()
print(" global scope:", spam)
    
'''
output:
local scope: spam
 nonlocal scope: nonlocal spam
 global scope: nonlocal spam  
 global scope: global spam 
 '''
          
