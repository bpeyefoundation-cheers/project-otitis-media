class Person:
    def __init__(self, firstname, lastname, age, country, city):
        # TODO: Assign the argument firstname to the attribute firstname of the class Person and so on.
        self.firstname= firstname
        self.lastname = lastname
        self.age = age
        self.country = country
        self.city = city
        



# TODO: Pass your firstname and lastname arguments as strings, age as a number, country and city as strings
p = Person('Sonu', 'G.C', 22 ,'Nepal' , 'Kathmandu')

# TODO: ends here

print(p.firstname)
print(p.lastname)
print(p.age)
print(p.country)
print(p.city)