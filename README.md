# Deep-Cloud
Deep learning algorithm are known to be computationally expensive and can take several days to train depending upon the size of data-set. To speed up the processing time, the use of GPU and distributed computing using map reduce can be seen. In this project we try to combine both of these processing paradigm. We use map reduce to implement a neural network. Each mapper or individual machine is equipped with a GPU and uses Theano/Tensorflow for GPU multi-threading. Further, the reducer use genetic algorithms to speed up the convergence rate with the help of 2 roulette selection. The project has been implemented on neural networks but can be extended to other deep neural models.
