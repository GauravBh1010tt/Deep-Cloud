def reducer(self,key,value):
        cost = sum(value)
        #print 'reducer ', key,'  ',2*float(cost)/(m)
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
