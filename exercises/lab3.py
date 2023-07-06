
# Understanding the default arguments in Python

# Sometimes, you may want to have a default values for your object methods. If we give default values for the parameters in the constructor, we can avoid errors when we call or instantiate our class without parameters. Let's see how it looks:

# TODO: Create a class called Person with default arguments of yours
class Person:
      def __init__(self, firstname='Manish', lastname='Dhakal', age=24, country='Finland', city='Helsinki'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'


p1 = Person()
print(p1.person_info())

# TODO:
# Pass your friends details here
p2 = Person('John', 'Doe', 30, 'Nomanland', 'Noman city')
print(p2.person_info())