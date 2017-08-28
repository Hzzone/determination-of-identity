Test 20 dicoms, 2 of per person on LeNet.   
Since the limitation of Google protobuf, its max buffer size is only 64MB, but the 3D CT images of head are too big to accommadate. So the maximum number of slices of each sample is 150.

Pay attention to it is trained on the same dataset.
[trained model download.](http://omoitwcai.bkt.clouddn.com/lenet_iter_10000.caffemodel)     
Test on 2 more person sample, cosine_distance 0.978557.  
on different person sample, 0.748577.

<div align=center>
<img src= "http://omoitwcai.bkt.clouddn.com/2017-08-28-Figure_1.png"/>
</div>

------
Lenet siamese network.      



```
[libprotobuf ERROR google/protobuf/io/coded_stream.cc:180] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
```

 

