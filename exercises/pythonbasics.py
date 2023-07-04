#printing the numbers from th list 
#list= [1,2,3,4,5]
#for i in list:
 #   print(i)

#priting the range of numbers 
#for i in range(5):
 #   print(i)

#iterates through the numbers until the value 3 is reached 
'''for i in range(5):
    if i == 3:
        break 

    print(i)'''

# iterates through the given range of numbers except when i==3
'''for i in range(5):
    if i == 3:
        continue
    else:
        print(i)'''

#printing the pattern 
# n=5
# for i in range(1, n+1):
#     for j in range(1, i+1):
#         print( j, end=' ')
#     print('')

#printing the random items from the list 
'''list= [1,2,3,4,5,6,7,8,9]
a=len(list)
print(list[0])
print(list[1])
print(list[a-1])
print(list[a-2])'''


# store the first 5 elements of the list into a new variable and print the new variable 

# list= [1,2,3,4,5,6,7,8,9]
# new_list=[]
# for i in range(0,5):
#     new_list.append(list[i])

# print(new_list)


## store the last 6 elements a into a new variable and print the new variable
# list= [1,2,3,4,5,6,7,8,9]
# new_list=[]
# length= len(list)
# print(length)
# for i in range(length-1,3,-1):
#     new_list.append(list[i])

# print(new_list)


# print every second element of the list a
# list= [1,2,3,4,5,6,7,8,9]
# for i in range(0, len(list), 2):
#     print(list[i])

#if a = [1,2,3,4,5], then required solution is [2, 4] 
# a= [1,2,3,4,5]
# a1=[]
# for i in range(1, len(a)-1, 2):
#     a1.append(a[i])
# print(a1)

#print length of the list a
# a= [1,2,3,4,5,6,7]
# print(len(a))

#creating the new list 
'''list=[]
for i in range(1, 11):
    list.append(i**2)
print(list)'''    


# create a new list consisting of squareroot of each element of list a
'''import math
a= [1,4,9,16,25,36,49]
new_list=[]
for i in range(0, len(a)-1):
    new_list.append(math.sqrt(a[i]))

print(new_list)'''
 

#write one line python code that makes a new list containing only even numbers in list a
'''a= [1,2,3,4,5,6,7]
new_list=[]
for i in range(0, len(a)):
    if(a[i] % 2 == 0):
        new_list.append(a[i])

print(new_list)'''












#list comprehension in python 
'''a=[i**2 for i in range(11) if i%2 ==0]
print(a)

print(a[:-2:2])'''




# project_dict = {'otitis_media':['prasanna', 'sonu', 'manvi'], 
#                  'oral_cancer': ['rahul', 'biru','manish'],
#                  'genetic_disorder': ['ayush', 'ayushma' ,'bhim', 'rusha']
#                }


# print(project_dict["otitis_media"])

# this dictionary contains incorrect project assignment, create a new dictionary with correct assignment of project
'''project_dict = {'otitis_media':['prasanna', 'sonu', 'manvi'], 
                 'oral_cancer': ['rahul', 'biru','manish'],
                 'genetic_disorder': ['ayush', 'ayushma' ,'bhim', 'rusha']
               }


project_dict.update({"otitis_media" : ['manish' , 'sonu', 'ayushma'],
                     "oral_cancer": ['prasanna', 'biru', 'rahul'],
                       "genetic_disorder" :['aayush', 'manvi', 'rusha']  })

print(project_dict)'''


#incrementing the value of the doctinary 
bp_eye_daily_patient_enrollment = {
    'ENT': 100,
    'dental': 50,
    'eye': 200 
}

# for x in bp_eye_daily_patient_enrollment.keys():
#     bp_eye_daily_patient_enrollment[x] += 50

# print(bp_eye_daily_patient_enrollment)
     
 

