import numpy as np
# def calc_weight_sum(input:np.array,weight:np.array,intercept:float):

#     weighted_sum=np.dot(input,weight)+intercept
#     return weighted_sum

# def perceptron_output(input,weight,intercept):
#     weighted_sum=calc_weight_sum(input,weight,intercept)
#     percep_output=np.where(weighted_sum<0,-1,1)
#     return percep_output



# def perceptron_learning(input,weight,intercept,target):
#     output=perceptron_output(input,weight,intercept)
#     error=np.where(target!=output,1,0)*target
#     updated_weight=np.dot(input,error)+weight
#     updated_intercept=intercept+target
#     # print("Input",input,"Out",output,"Error",error,"target",target)
#     return updated_weight,updated_intercept 


# if __name__=="__main__":
#     input=np.array([[1,2],[3,4],[5,6],[4,5]])
#     weight=np.array([0,0])
#     intercept=0
#     target=np.array([1,1,-1,-1])

#     epochs=10
#     for e in range(epochs):
#         for x,y in zip(input,target):
       
#             weight, intercept =perceptron_learning(x,weight,intercept,y)
#         print("updated_result:", weight  ,intercept)

# target_value=perceptron_output(input[0],weight,intercept)
# print(target_value)

#using class
class Perceptron:
    def __init__(self,input) :
        self.weight=np.array([0,0])
        self.intercept=0
        
    def calc_weight_sum(self,input:np.array):
    
        weighted_sum=np.dot(input,self.weight)+self.intercept
        return weighted_sum
    def perceptron_output(self,input):
        weighted_sum=self.calc_weight_sum(input)
        percep_output=np.where(weighted_sum<0,-1,1)
        return percep_output
    def perceptron_learning(self,input,target,epochs):
        for _ in range(epochs):
            for x, y in zip(input, target):
                output = self.perceptron_output(x)
                error = np.where(y != output, 1, 0) * y
                self.weight += x * error
                self.intercept += y
if __name__ == "__main__":
        input_data = np.array([[1, 2], [3, 4], [5, 6], [4, 5]])
        target = np.array([1, 1, -1, -1])

        perceptron = Perceptron(input_data)

        epochs = 10
        perceptron.perceptron_learning(input_data, target, epochs)

        print("Updated weight:", perceptron.weight)
        print("Updated intercept:", perceptron.intercept)