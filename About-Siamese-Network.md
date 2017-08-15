#### 参考文献
[Siamese Network Training with Caffe](http://caffe.berkeleyvision.org/gathered/examples/siamese.html)   
[Github Siamese Network Training with Caffe](https://github.com/BVLC/caffe/tree/master/examples/siamese)    
[Caffe中的Siamese网络](https://vra.github.io/2016/12/13/siamese-caffe/)     
[convert_mnist_siamese_data.cpp](https://github.com/BVLC/caffe/blob/master/examples/siamese/convert_mnist_siamese_data.cpp)     
[基于2-channel  network的图片相似度判别](http://blog.csdn.net/hjimce/article/details/50098483)    
![Caffe Mnist Siamese Network Example](http://omoitwcai.bkt.clouddn.com/editor.jpg) 


#### 个人理解

##### 数据处理
在[convert_mnist_siamese_data.cpp](https://github.com/BVLC/caffe/blob/master/examples/siamese/convert_mnist_siamese_data.cpp)中描述了转化成pair_data的具体过程.  
对于mnist数据集来说,是一个灰度图,根据tutorial描述的和相关源码的阅读,转化成pair_data是这样的: 
```
We start with a data layer that reads from the LevelDB database we created earlier. Each entry in this database contains the image data for a pair of images (pair_data) and a binary label saying if they belong to the same class or different classes (sim).
```
也就是说在LevelDB中的存储形式是一个样本是成对存在的,通过设置一个二进制位来表示是否是同一个,例如0表示不是同一个,1则相反.  
又通过以下生成Caffe中的生成LevelDB的一段关键代码:
```C++
  for (int itemid = 0; itemid < num_items; ++itemid) { //这里是所有样本.txt
    int i = caffe::caffe_rng_rand() % num_items;  // pick a random  pair
    int j = caffe::caffe_rng_rand() % num_items; //然后在样本中随机取两个
    read_image(&image_file, &label_file, i, rows, cols,
        pixels, &label_i);
    read_image(&image_file, &label_file, j, rows, cols,
        pixels + (rows * cols), &label_j); // 分别读出来, 然后一起存到一个2*rows*cols的矩阵里面
    datum.set_data(pixels, 2*rows*cols); // 意味着一对数据是存在一起的
    if (label_i  == label_j) {   // 在train.txt或者test.txt中分别是文件(空格)标签形式表示
      datum.set_label(1);       // 每一行代表一个样本,同一类的标签相同
    } else {                    // 在这里就通过比对标签,然后设置上面提到的位来表示是否是同一类
      datum.set_label(0);
    }
    datum.SerializeToString(&value);
    std::string key_str = caffe::format_int(itemid, 8);
    db->Put(leveldb::WriteOptions(), key_str, value); //最后这里就保存到LevelDB中
  }
```
数据处理大概是这样的.

##### 网络结构
Caffe有一个pair_data层表示成对数据
```
layer {
  name: "pair_data"
  type: "Data"
  top: "pair_data"
  top: "sim"
  include { phase: TRAIN }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/siamese/mnist_siamese_train_leveldb"
    batch_size: 64
  }
}
```
在Siamese Network中,两个子网络是完全一样的,但是在data层之后通过一个slice层将这一对数据分开:
```
layer {
  name: "slice_pair"
  type: "Slice"
  bottom: "pair_data"
  top: "data"
  top: "data_p" //从这里data层数据就被切开了
  slice_param {
    slice_dim: 1 //这里应该是分开的维度,比如mnist是一个2*rows*cols的,所以这里是1
    slice_point: 1
  }
}
```
通过上面的方法就讲两张图片分开了.
```
Naming the parameters allows Caffe to share the parameters between layers on both sides of the siamese net. 
```
通过多复制一个convolutional and inner product layers来达到两个子网络共享参数, 所以在Siamese Network中或出现两个分支,都有参数的情况.  
举个例子, 在Mnist的示例中conv1是这样定义的:
```
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    name: "conv1_w"
    lr_mult: 1
  }
  param {
    name: "conv1_b"
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```
其实子网络我认为和普通的卷积神经网络是一样的,而且siamese network中两个子网络是一模一样, 但是最重要的是:
*两个分支提取特征的过程是独立的.    
[基于2-channel  network的图片相似度判别](http://blog.csdn.net/hjimce/article/details/50098483)这篇文章里描述的方法其他和传统的siamese一样,只是在子网络这层,近似于没有分支了,把两个patch打包在一起,然后输进网络,所以才会有文章中描述的图片:     
![](http://img.blog.csdn.net/20151204201146814?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
![](http://img.blog.csdn.net/20151204202905335?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
左边是siamese,右边是文章描述的方法.  

最后是loss层,Caffe中有一个Contrastive Loss Function, 具体实现则是[```CONTRASTIVE_LOSS layer```](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ContrastiveLossLayer.html)

![](http://caffe.berkeleyvision.org/doxygen/form_51.png)    
![](http://caffe.berkeleyvision.org/doxygen/form_52.png)
```
layer {
    name: "loss"
    type: "ContrastiveLoss"
    contrastive_loss_param {
        margin: 1.0
    }
    bottom: "feat"
    bottom: "feat_p"
    bottom: "sim"
    top: "loss"
}
```

##### 效果
在示例里面没有输出相似度,而是聚类的效果:

![](http://omoitwcai.bkt.clouddn.com/Screenshot%20from%202017-08-16%2004:20:07.png)

// TODO 输出相似度,并做一个网络的accuracy输出
