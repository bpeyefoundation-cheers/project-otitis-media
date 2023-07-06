
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
