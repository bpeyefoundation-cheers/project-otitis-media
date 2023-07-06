#lab 2
class Person:
      def __init__(self, firstname, lastname, age, country, city):
        self.firstname = firstname
        self.lastname = lastname
        self.age = age
        self.country = country
        self.city = city

      def person_info(self):
           #return self.firstname,self.lastname,self.age,self.country,self.city
        return f"{self.firstname} {self.lastname} is  {self.age} years old . She lives in {self.country} {self.city}"

a=Person("Ayushma","Pandey",10,"Nepal","Lalitpur")
print(a.person_info())


