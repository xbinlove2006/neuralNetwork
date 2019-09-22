from neuralnetwork import NeuralNetwork
import numpy
import scipy.special
import matplotlib.pyplot
import time as date
inputnodes=784
hiddennodes1=200
hiddennodes2=100
outputnodes=10
learningrate=0.2
n=NeuralNetwork(inputnodes,hiddennodes1,hiddennodes2,outputnodes,learningrate)
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
print("4层神经网络:\n隐藏层1节点数：",hiddennodes1,"隐藏层2节点数",hiddennodes2,"\n用时",end-start)
print("准确率：")
print(score.sum()/score.size)