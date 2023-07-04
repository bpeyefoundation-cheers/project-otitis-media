# write a function called 'add' that takes two numbers as arguments

# def add(x, y):
#     sum= x+y
#     return sum
# sum = add(5,6)
# print(sum)

#list

first_list=[1,2,3,4,5]
def create_list(first_list):
    new_list=[i*2 for i in first_list]
    return new_list

double_valued_list = create_list(first_list)  
print(double_valued_list)  