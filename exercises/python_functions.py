#defining add function
def add(a,b):
    c=a+b
    return c
s=add(1,2)
print(s)
#creating list
def create_lst(lst):
    emp_lst=[]
    for e in lst:
        
        emp_lst.append(e*2)


    return emp_lst
double_lst=create_lst([1,2,3,4,5])
print(double_lst)

a=[e*2 for e in [2,4,6,8]] 
print(a)
#adding list
def add_lst(l1,l2):
    output_lst=[]
    for i,j in zip(l1,l2):
        sum=i+j
        output_lst.append(sum)
                         
    return output_lst
sum_list=add_lst([1,2,3,4],[2,4,6,7])
print(sum_list)

#printing _number
def print_pattern(num):
    for i in range(1,num+1):
        for j in range(1,i+1):                                              
            print(j,end=' ')                                                  
        print()

print(print_pattern(6))
                                                                                                                                                                              



    
