import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetwork:
    def __init__(self,inputnodes,hiddennodes1,hiddennodes2,outputnodes,learningrate):
        self.innodes=inputnodes
        self.hinodes1=hiddennodes1
        self.hinodes2=hiddennodes2
        self.ounodes=outputnodes
        self.rate=learningrate
        self.w_in_hi=numpy.random.rand(self.hinodes1,self.innodes)-0.5
        self.w_hi1_hi2=numpy.random.rand(self.hinodes2,self.hinodes1)-0.5
        self.w_hi_ou=numpy.random.rand(self.ounodes,self.hinodes2)-0.5
        self.act=lambda x:scipy.special.expit(x)
        pass
    def train(self,input_list,target_list):
        inputs=numpy.array(input_list,ndmin=2).T
        targets=numpy.array(target_list,ndmin=2).T

        hi1_inputs=numpy.dot(self.w_in_hi,inputs)
        hi1_outputs=self.act(hi1_inputs)

        hi2_inputs=numpy.dot(self.w_hi1_hi2,hi1_outputs)
        hi2_outputs=self.act(hi2_inputs)

        fi_inputs=numpy.dot(self.w_hi_ou,hi2_outputs)
        fi_outputs=self.act(fi_inputs)

        fi_error=targets-fi_outputs
        hi2_error=numpy.dot(self.w_hi_ou.T,fi_error)
        hi1_error=numpy.dot(self.w_hi1_hi2.T,hi2_error)
        self.w_hi_ou+=self.rate*numpy.dot((fi_error*fi_outputs*(1-fi_outputs)),numpy.transpose(hi2_outputs))

        self.w_hi1_hi2+=self.rate*numpy.dot((hi2_error*hi2_outputs*(1-hi2_outputs)),numpy.transpose(hi1_outputs))

        self.w_in_hi+=self.rate*numpy.dot((hi1_error*hi1_outputs*(1-hi1_outputs)),numpy.transpose(inputs))
        pass
    def query(self,input_list):
        inputs=numpy.array(input_list,ndmin=2).T
        hi1_inputs=numpy.dot(self.w_in_hi,inputs)
        hi1_outputs=self.act(hi1_inputs)

        hi2_inputs=numpy.dot(self.w_hi1_hi2,hi1_outputs)
        hi2_outputs=self.act(hi2_inputs)

        fi_inputs=numpy.dot(self.w_hi_ou,hi2_outputs)
        fi_outputs=self.act(fi_inputs)
        return fi_outputs
        pass
    pass
