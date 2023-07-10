# Method to Modify Class Default Values
"""
In the example below, the person class, all the constructor parameters have default values. 
In addition to that, we have skills parameter, which we can access using a method. 
Let us create add_skill method to add skills to the skills list.
"""

class Person:
      def __init__(self, firstname='Asabeneh', lastname='Yetayeh', age=250, country='Finland', city='Helsinki'):
          self.firstname = firstname
          self.lastname = lastname
          self.age = age
          self.country = country
          self.city = city
          self.skills = []

      def person_info(self):
        return f'{self.firstname} {self.lastname} is {self.age} years old. He lives in {self.city}, {self.country}.'
      def add_skill(self, skill):
        # TODO: add the skill passed as an argument to the list of skills
        # Hint: use the "append" method to add an item to a list
        self.skill=skill
        self.skills.append(skill)

    
    
         
      
p1 = Person()
print(p1.person_info())
#print(p1.add_skill('driving','coo'))
print(p1.skills)

# TODO: Using the add_skill method, add your 5 major skills


p2 = Person('John', 'Doe', 30, 'Nomanland', 'Noman city')
print(p2.person_info())
p2.add_skill('driving')
print(p2.skills)
p2.add_skill('cooking')
print(p2.skills)