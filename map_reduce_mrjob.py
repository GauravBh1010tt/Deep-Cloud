# -*- coding: utf-8 -*-

import math
import csv
import theano
import numpy as np
import theano.tensor as T
import numpy.random as rn
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env

data = []
with open('data.csv') as handle:
    a=csv.reader(handle)
    for i in a:
        data.append(map(float,i))
        
inp_size = 400
hidden_size = 25
out_size = 10
reg = 0.01
alpha = 0.01
m = 50
dic = [10]

final_cost = []
parent = {}

w1 = theano.shared(rn.randn(inp_size,hidden_size).astype('float32'),name='w1')
w2 = theano.shared(rn.randn(hidden_size,out_size).astype('float32'),name='w2')
b1 = theano.shared(np.zeros(hidden_size).astype('float32'),name='b1')
b2 = theano.shared(np.ones(out_size).astype('float32'),name='b2')

class MRHadoopFormatJob(MRJob):
    
    def init_mapper(self):
        self.count = 1
        self.gradients = []
        self.final_cost = []

        #x = theano.shared(name = 'x')
        #y = theano.shared(name = 'y')
        self.x = theano.shared(np.matrix(np.zeros((inp_size,1))).astype('float32'),'x') # matrix of doubles
        self.y = theano.shared(np.matrix(np.zeros((out_size,1))).astype('float32'),'y') # vector of int64
        a = T.tanh(self.x.dot(w1)+b1)
        h = T.nnet.softmax(a.dot(w2)+b2)    
        #print 'here'
        #err = -y*T.log(h) - (1-y)*T.log(1-h)
        #cost = err.mean() + 1./m * reg/2 * ((w1**2).sum()+(w2**2).sum())        
        loss_reg = 1./m* reg/2 * ((w1**2).sum()+(w2**2).sum())
        cost = T.nnet.categorical_crossentropy(h,self.y).mean()*(2./m) +loss_reg
        pred = T.argmax(h,axis=1)
        #print 'here'
        gw1 = T.grad(cost,w1)
        gw2 = T.grad(cost,w2)
        gb1 = T.grad(cost,b1)
        gb2 = T.grad(cost,b2)
        #self.forward_prop = theano.function([],h)
        self.compute_cost = theano.function([],outputs=[cost,gw1,gw2,gb1,gb2])
        #self.predict = theano.function([],pred)
    
    #OUTPUT_PROTOCOL = 'PickleProtocol'
    def mapper1(self, key, value):

        value = map(float,value.split(','))
        #print 'mapper  ','  ',key,'    ',value
        x_train = np.array(value[:-1])
        y_train = np.zeros(out_size)
        y_train[value[-1]]  = 1      
        #print x_train
        #print y_train
        y_train = np.matrix(y_train)
        
        x_train = np.matrix(x_train)
        self.x.set_value(x_train.astype('float32'))
        self.y.set_value(y_train.astype('float32'))
        
        grads = self.compute_cost()
        #print 'here'
        b = jobconf_from_env('mapreduce.task.partition')
        #a = np.asarray(grads[3])
        #print a
        if self.count % 50 == 0:
            #b = jobconf_from_env('mapreduce.task.partition')
            print 'cost is ',float(grads[0]),'  mapper',b,' iteration :: ', self.count
            #dic[1] = grads
            #cost_all.append((b,cost))
        if len(self.gradients) == 0:
            self.gradients = grads
        else:
            for i in range(0,5):
                self.gradients[i] += grads[i]
        #c = np.matrix(np.zeros((32,25)))
        #c = range(1,500)

        self.count+=1
        yield b, float(grads[0])
             
    def mapper2(self, key, value):
        #print key,value
        x_train = np.array(key[:-1])
        y_train = np.zeros(out_size)
        y_train[key[-1]]  = 1 
        
        y_train = np.matrix(y_train)
        y_train = np.matrix(y_train)
        x_train = np.matrix(x_train)
        
        self.x.set_value(x_train.astype('float32'))
        self.y.set_value(y_train.astype('float32'))
        #predict = theano.function([],pred)
        b = jobconf_from_env('mapreduce.task.partition')
        grads = self.compute_cost()
        if self.count % 50 == 0:
            #b = jobconf_from_env('mapreduce.task.partition')
            print 'cost is ',float(grads[0]),'  mapper',b,' iteration :: ',self.count
        if len(self.gradients) == 0:
            self.gradients = grads
        else:
            for i in range(0,5):
                self.gradients[i] += grads[i]
        self.count+=1
        yield b,float(grads[0])
  
    def reducer(self,key,value):
        cost = sum(value)
        print 'reducer ', key,'  ',2*float(cost)/(m)
        temp1 = parent['0']
        temp2 = parent['1']
        all_weights = []

        for i in range(1,3):
            mid = temp1[i].shape[0] / 2
            temp3 = np.matrix(np.zeros((temp1[i].shape)))
            temp3 [0:mid,:] = np.matrix(temp1[i])[0:mid,:]
            temp3 [mid:,:] = np.matrix(temp2[i])[mid:,:]
            all_weights.append(temp3)

        w1.set_value(all_weights[0].astype('float32'))
        w2.set_value(all_weights[1].astype('float32'))
        #print 'here',dic['b1'].tolist()[0],b1.get_value()
        b1.set_value((np.asarray(temp1[3])).astype('float32'))
        b2.set_value((np.asarray(temp2[4])).astype('float32'))
                            
        for i in data:
            yield i,1

    def final_mapper(self):
        print '........in final mapper...............',(self.gradients[0] / self.count)
        final_cost.append(self.gradients[0] / self.count)
        #print 'cost after iteration ',self.count,' is ',(self.gradients[0] / self.count)
        temp_w1 = w1.get_value()
        temp_w2 = w2.get_value()
        temp_b1 = b1.get_value()
        temp_b2 = b2.get_value()
        #dic.append(1)
        #print 'dic is ',dic[0]
        b = jobconf_from_env('mapreduce.task.partition')
        print 'now ',b
        self.gradients[1] = temp_w1 - (alpha*self.gradients[1])
        self.gradients[2] = temp_w2 - (alpha*self.gradients[2])
        self.gradients[3] = temp_b1 - (alpha*np.asarray(self.gradients[3]))
        self.gradients[4] = temp_b2 - (alpha*np.asarray(self.gradients[4]))
        parent[b] = self.gradients
        '''w1.set_value(self.gradients[1].astype('float32'))
        w2.set_value(self.gradients[2].astype('float32'))
        #print 'here',dic['b1'].tolist()[0],b1.get_value()
        b1.set_value((np.asarray(self.gradients[3])).astype('float32'))
        b2.set_value((np.asarray(self.gradients[4])).astype('float32'))'''
        self.gradients = []
        #dic[0]+=100
               
    def steps(self):
        return [ MRStep(mapper_init=self.init_mapper,mapper=self.mapper1,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),
 MRStep(mapper_init=self.init_mapper,mapper=self.mapper2,reducer=self.reducer,mapper_final=self.final_mapper),]


if __name__ == '__main__':
    MRHadoopFormatJob.run()