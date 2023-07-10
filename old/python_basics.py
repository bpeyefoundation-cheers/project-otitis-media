#printing number from list
for i in [1,2,3,4,5]:

    print(i)

for i in range(5):
        print(i)
for i in range(5):
      if i==3:
        break
      print(i)
for i in range(5):
     if i==3:
          continue
     print(i)
for i in range(5):
     if i==3:
          continue
     else:
          print(i)

for i in range(1,6):
     for j in range(1,i+1):
          print(j,end=' ')
     print()

a=[1,4,9,16,25,36,49,64,81,100]
print(a[0])
print(a[-1])
print(a[-2])

b=a[:5]
print(b)

c=a[-6:]
print(c)

print(len(a))

print(a[1::2])
d=[]
for element in a:
     c=element*element
     d.append(c)
print(d)
l=[]
import math
for element in d:
    f=math.sqrt(element)
    l.append(f)
print(l)
# list comprehension
a=[i for i in range(10) if i%2==0]
print(a)
a=[(i,j) for i in range(10) for j in range(i)]
print(a)

project_dict={ 
     "otitis_media_member":["ayushma","sonu","manish"],
     "oral_member":["biru","prasanna"]
     }
print(project_dict["otitis_media_member"][0])
bp_eye_daily_patient_enrollment = {
    'ENT': 100,
    'dental': 50,
    'eye': 200 
}
for patient in bp_eye_daily_patient_enrollment:
     bp_eye_daily_patient_enrollment[patient]+=50
print(bp_eye_daily_patient_enrollment)

for k in bp_eye_daily_patient_enrollment.keys():
     bp_eye_daily_patient_enrollment[k]=bp_eye_daily_patient_enrollment[k]+10
print(bp_eye_daily_patient_enrollment)


     
     
      



