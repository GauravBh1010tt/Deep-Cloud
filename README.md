# Deep-Cloud
###PLease refer to my blog [Deep Cloud](https://deeplearn.school.blog/2016/12/24/deep-cloud/#more-238) for the implementation details.

Deep learning algorithm are known to be computationally expensive and can take several days to train depending upon the size of data-set. To speed up the processing time, the use of GPU and distributed computing using map reduce can be seen. In this project I have tried to combine both of these processing paradigm. 

### Features of the project:

- Mrjob is used as a MapReduce abstraction to implement a two layer neural network. 
- Each mapper or individual machine is equipped with a GPU and uses Theano/Tensorflow for GPU multi-threading. 
- Gradients are computed in the mapper throught backpropagation.
- The reducer performs the updation of weights.
- The number of epochs is the number of time the `step` function is called.
