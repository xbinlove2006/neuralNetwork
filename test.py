import numpy
import scipy.special
import matplotlib.pyplot
import time as date

class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.innodes=inputnodes
        self.hinodes=hiddennodes
        self.ounodes=outputnodes
        self.rate=learningrate
        self.w_in_hi=numpy.random.rand(self.hinodes,self.innodes)-0.5
        self.w_hi_ou=numpy.random.rand(self.ounodes,self.hinodes)-0.5
        self.act=lambda x:scipy.special.expit(x)
        pass
    def train(self,input_list,target_list):
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(target_list,ndmin=2).T
        hi_inputs=numpy.dot(self.w_in_hi,inputs)
        hi_outputs=self.act(hi_inputs)
        fi_inputs=numpy.dot(self.w_hi_ou,hi_outputs)
        fi_outputs=self.act(fi_inputs)
        fi_error=targets-fi_outputs
        hi_error=numpy.dot(self.w_hi_ou.T,fi_error)
        self.w_hi_ou+=self.rate*numpy.dot((fi_error*fi_outputs*(1-fi_outputs)),numpy.transpose(hi_outputs))
        self.w_in_hi+=self.rate*numpy.dot((hi_error*hi_outputs*(1-hi_outputs)),numpy.transpose(inputs))
        pass
    def query(self,input_list):
        inputs=numpy.array(input_list,ndmin=2).T
        hi_inputs=numpy.dot(self.w_in_hi,inputs)
        hi_outputs=self.act(hi_inputs)
        fi_inputs=numpy.dot(self.w_hi_ou,hi_outputs)
        fi_outputs=self.act(fi_inputs)
        return fi_outputs
        pass
    pass
inputnodes=784
hiddennodes=300
outputnodes=10
learningrate=0.2
n=NeuralNetwork(inputnodes,hiddennodes,outputnodes,learningrate)
data_file=open("mnist/mnist_train.csv","r")
data_file_list=data_file.readlines()
data_file.close()
start=date.time()
for list in data_file_list:
    all_values=list.split(',')
    index=int(all_values[0])
    inputs=numpy.asfarray(all_values[1:])/255.0*0.99+0.01
    targets=numpy.zeros(outputnodes)+0.01
    targets[index]=0.99
    n.train(inputs,targets)
    pass
data_file=open("mnist/mnist_test.csv","r")
data_file_list=data_file.readlines()
data_file.close()
score=numpy.zeros(len(data_file_list))
i=0
for list in data_file_list:
    all_values=list.split(',')
    label=int(all_values[0])
    inputs=numpy.asfarray(all_values[1:])/255.0*0.99+0.01
    targets=n.query(inputs)
    index=numpy.argmax(targets)
    if index==label:
        score[i]=1
    else:
        score[i]=0
    i+=1
end=date.time()
print("3层神经网络:\n隐藏层节点数：",hiddennodes,"\n用时",end-start)
print("准确率：")
print(score.sum()/score.size)
