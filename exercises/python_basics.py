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


     
     
      



