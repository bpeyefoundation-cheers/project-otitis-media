class Person:
      def __init__(self, firstname, lastname, age, country, city):
        self.firstname = firstname
        self.lastname = lastname
        self.age = age
        self.country = country
        self.city = city

      def person_info(self):

        # TODO: Return string of the following format
        # {firstname} {lastname} is {age} years old. He lives in {city}, {country}
        return f"{self.firstname} {self.lastname}  is {self.age} years old. He lives in {self.country} {self.city}" 

# TODO: Pass your firstname and lastname arguments as strings, age as a number, country and city as strings
p = Person("sonu" , "gc" , 22, "nepal", "ktm")
print(p.person_info())