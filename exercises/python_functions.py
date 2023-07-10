# write a function called 'add' that takes two numbers as arguments

# def add(x, y):
#     sum= x+y
#     return sum
# sum = add(5,6)
# print(sum)

# write a function that takes a list as argument and returns another list where each element of the new list is double

# first_list=[1,2,3,4,5]
# def create_list(first_list):
#     new_list=[i*2 for i in first_list]
#     return new_list

# double_valued_list = create_list(first_list)  
# print(double_valued_list)  


# write a function that takes two lists as argument and returns another list where each element of the new list is the sum of corresponding element in the given lists

# list1 = [1,2,3,4,5]
# list2 = [6,7,8,9,10]
# def add_lists(list1, list2):
#     add_elements = [list1[i] + list2[i] for i in range(0, len(list1))]
#     return add_elements

# new_list = add_lists(list1, list2)
# print(new_list)



#alternative1 using zip

# list1 = [1,2,3,4,5]
# list2 = [6,7,8,9,10]
# def add_lists(list1, list2):
#     add_elements = [a+b for a, b in zip(list1, list2)]  #zip creates the tuple of correspoding elements
#     return add_elements

# new_list = add_lists(list1, list2)
# print(new_list)


# write a function that takes a integer as argument and prints the pattern as shown below

# def print_pattern(num) :
#     for i in range(1, num+1):
#         for j in range(1, i+1):
#             print(j, end= ' ')
#         print( )

# print_pattern(10)



