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
            #print 'cost is ',float(grads[0]),'  mapper',b,' iteration :: ', self.count
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
