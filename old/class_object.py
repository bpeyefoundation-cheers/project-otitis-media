## Class Constructor
"""
In the examples above, we have created an object from the Person class. However, a class without a constructor is not really useful in real applications. 
Let us use constructor function to make our class more useful. Like the constructor function in Java or JavaScript, 
Python has also a built-in init() constructor function. 
The __init__ constructor function has self parameter which is a reference to the current instance of the class Examples:
"""

class Person:
      def __init__ (self, name):
        # self allows to attach parameter to the class
          self.name =name

# TODO: Replace with your name
p = Person('Ayushma')
print(p.name)
print(p)

q=Person('Ram')
print(q.name)
print(q)
#lab1
class Person:
    def __init__(self, firstname, lastname, age, country, city):
        # TODO: Assign the argument firstname to the attribute firstname of the class Person and so on.
        self.firstname=firstname
        self.lastname=lastname
        self.age=age
        self.country=country
        self.city=city
a=Person('Ayushma','Pandey',30,'Nepal','Lalitpur')
print(a.firstname)
print(a.lastname)
print(a.age)
print(a.country)
print(a.city)
print(a)