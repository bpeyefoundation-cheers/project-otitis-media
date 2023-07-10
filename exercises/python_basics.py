#printing the numbers from 1 to 5
#for i in [1,2,3,4,5]:
#   print(i)

#printing the number from 0 to 5   
#for i in range(5):
 #   print(i)

# for i in range(5):
#     if i==3:
#         break
#     print(i)

# for i in range(5):
#     if i==3 :
#         continue
#     print(i)


# for i in range(5):
#     if i==3 :
#         continue
#     else :
#         print(i)


# for i in range(5):
#     if i== 6 :
#         continue
#     else :
#         print(i)

#printing pattern
# for i in range(1,6):
#     for j in range(1, 1+i):
#         print(j, end=' ') 
#     print(' ')

#print list
# a=[1,2,3,4,5]
# print(a)


#print list using loop
# a = [i**2 for i in range(10)]
# print(a[::2])

# a=[]
# for i in  range(1, 11):
#     a.append(pow(i,2))
# print(a)
# print(a[0])
# print(a[1])
# print(a[9])
# print(a[-2])
# print(len(a))

## store the first 5 elements of a into a new variable and print the new variable 
# b=[]
# b.append(a[0:5])
# print(b)


# create a dictionary with the keys

bp_eye_daily_patient_enrollment = {
    'ent': 100,
    'dental' : 50,
    'eye' : 200
}


# increase all the enrolled patient  in patient enrollment dict by 50
# for patient in bp_eye_daily_patient_enrollment:
#     bp_eye_daily_patient_enrollment[patient]+= 50
# print(bp_eye_daily_patient_enrollment)


#alternative
# for k, v in bp_eye_daily_patient_enrollment.items():
#     bp_eye_daily_patient_enrollment[k] = v+50
# print(bp_eye_daily_patient_enrollment)

#using keys
# for k in bp_eye_daily_patient_enrollment.keys():
#     bp_eye_daily_patient_enrollment[k] = bp_eye_daily_patient_enrollment[k]+50
# print(bp_eye_daily_patient_enrollment)





