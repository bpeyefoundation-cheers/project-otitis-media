# write a function called 'add' that takes two numbers as arguments

'''def add(num1, num2):
    return num1+num2

sum= add(5,10)
print(sum)'''


# write a function that takes a list as argument and returns another list
# where each element of the new list is double of the given list

'''list_of_num=[1,2,3,4]
def double_values(list_1):
   new_list=[2*x for x in list_1]
   return new_list

new_list=double_values(list_of_num)
print(new_list)'''



# write a function that takes two lists as argument and returns another list where each 
# element of the new list is the sum of corresponding element in the given lists
'''l1=[1,2,3,4]
l2=[5,6,7,8]     
def sum_of_list(list1, list2):
    new_list=[list1[i]+list2[i] for i in range(0,len(list1))]
    return new_list

new_list=sum_of_list(l1,l2)
print(new_list )'''


# write a function that takes a integer as argument and prints the pattern as shown below
'''def pattern_print(num):
    for i in range(1, num+1):
        for j in range(1, i+1):
            print(j , end= ' ')
        print('')

pattern_print(5)'''