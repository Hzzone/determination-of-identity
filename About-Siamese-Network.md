## 参考文献
[Siamese Network Training with Caffe](http://caffe.berkeleyvision.org/gathered/examples/siamese.html)   
[Github Siamese Network Training with Caffe](https://github.com/BVLC/caffe/tree/master/examples/siamese)    
[Caffe中的Siamese网络](https://vra.github.io/2016/12/13/siamese-caffe/)     
[convert_mnist_siamese_data.cpp](https://github.com/BVLC/caffe/blob/master/examples/siamese/convert_mnist_siamese_data.cpp)     
[基于2-channel  network的图片相似度判别](http://blog.csdn.net/hjimce/article/details/50098483)    
[机器学习中的相似性度量](http://www.cnblogs.com/heaad/archive/2011/03/08/1977733.html)         
[人脸识别之caffe-face](http://blog.csdn.net/qq_14845119/article/details/53308996#reply)      
[caffe-face GitHub](https://github.com/ydwen/caffe-face)    
<div align=center>
<img src= "http://omoitwcai.bkt.clouddn.com/editor.jpg"/>
</div>

## 个人理解

### 数据处理
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

### 网络结构
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
**两个分支提取特征的过程是独立的**.    
[基于2-channel  network的图片相似度判别](http://blog.csdn.net/hjimce/article/details/50098483)这篇文章里描述的方法其他和传统的siamese一样,只是在子网络这层,近似于没有分支了,把两个patch打包在一起,然后输进网络,所以才会有文章中描述的图片:     
![](http://img.blog.csdn.net/20151204201146814?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
![](http://img.blog.csdn.net/20151204202905335?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
左边是siamese,右边是文章描述的方法.  

最后是loss层,Caffe中有一个Contrastive Loss Function, 具体实现则是[```CONTRASTIVE_LOSS layer```](http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1ContrastiveLossLayer.html).  
数学公式:
<div align=center>
<img src= "http://caffe.berkeleyvision.org/doxygen/form_51.png"/><br>
</div>
<div align=center>
<img src= "http://caffe.berkeleyvision.org/doxygen/form_52.png"/>
</div>

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

### 效果
在示例里面没有输出相似度,而是聚类的效果:

![](http://omoitwcai.bkt.clouddn.com/FvUMcEdVEaGzJETi3EtR9hbXYtzt)

// TODO 输出相似度,并做一个网络的accuracy输出

通过阅读[Caffe Python 分类示例](https://github.com/Hzzone/determination-of-identity/blob/master/mnist_siamese/mnist_siamese.ipynb)以及结合网络结构,可以很清楚知道,最后一层的输出其实是每一个样本在平面上的坐标,```(x, y)```一个二维向量,例如:
```
[[ 0.95238674 -1.03426003]
 [-1.46469998  0.09684839]
 [-1.1617192  -1.53521729]
 ..., 
 [ 1.65356219  1.25560427]
 [ 0.2914933   0.75894594]
 [-1.19767416  2.16366267]]
(10000, 2)
```
(10000, 2)则代表mnist数据集有10000个test样本,我直接输出的最后一层的shape.    
在平面上相聚越近的点则代表两个样本越相似.所以参考效果图,同一类的点越靠近.  
看一下LetNet的[train](https://github.com/Hzzone/determination-of-identity/blob/master/mnist_siamese/mnist_siamese_train_test.prototxt)和[deploy](https://github.com/Hzzone/determination-of-identity/blob/master/mnist_siamese/mnist_siamese.prototxt)文件之间的区别, 在测试的时候输入一个样本就行,而不需要两个样本合在一起.  
至于衡量网络的效果,除了loss收敛之外,可以看一下训练过程中的test loss, 还有更加准确的输出相似度.

在效果图上,为什么会出现聚类的效果,看一段关键代码:
```python
feat = out['feat']
f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
plt.show()
```
labels是label文件中每一个样本的标签,不同类的点颜色不同,所以才会出现这样的效果,通过example中的效果图可以清晰看到同一类的样本,靠的越近.  
例如这行代码是输出所有label为1的样本的坐标:
```python
print feat[labels==1]
print feat[labels==1].shape
```
我输出的结果是这样的:
```
[[-1.1617192  -1.53521729]
 [-1.17441916 -1.51869309]
 [-1.04867065 -1.4648124 ]
 ..., 
 [-1.0997206  -1.63687563]
 [-1.27621269 -1.66673696]
 [-1.22772038 -1.62820733]]
(1135, 2)
```
可以看到输出也是坐标,一共有1135个1的样本集,然后就明白效果图的含义了.
so....如何衡量相似度呢..可以有很多种方法,既然是坐标,那就算个欧式距离吧..其他的可以看[机器学习中的相似性度量](http://www.cnblogs.com/heaad/archive/2011/03/08/1977733.html).

我自己的测试代码在[siamese_test.py](https://github.com/Hzzone/determination-of-identity/tree/master/mnist_siamese/siamese_test.py)中.
```python
# 1 of test dataset
one = feat[labels==1]
# calculate euclidean distance
acc = np.sqrt(np.sum(np.square(one[0] - one[1])))
print acc
```
输出```0.0208408```

ps:
最后一层的```num_output=2```输出的是提取到的一个样本的特征向量,可以改的,并不是一定要2.  
通过以下代码输出每一层的输出的大小:

```python
# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
```
```(batch_size, channel_dim, height, width)```
```
data	(10000, 1, 28, 28)
conv1	(10000, 20, 24, 24)
pool1	(10000, 20, 12, 12)
conv2	(10000, 50, 8, 8)
pool2	(10000, 50, 4, 4)
ip1	(10000, 500)
ip2	(10000, 10)
feat	(10000, 2)
```

在这片文章里[人脸识别之caffe-face](http://blog.csdn.net/qq_14845119/article/details/53308996#reply),提到了用softmaxloss+softmaxloss来聚类,然后计算cosine distance代表相似度.
但是具体的相似度的值如何表示..只是计算特征向量的距离,我还不是很明白.

-------

如何去衡量训练的网络的准确度?我最近有了比较合理的思路,也是无意中看到一篇文章上面的.     
最后一层输出特征向量之后,选择和合适的距离函数,然后选择不同的阈值,在这个阈值下,网络的输出是否正确.     
比如讲相同的,在这个阈值上则为正确,否则错误,然后再加上不同的,就可以得到正确的/总数=识别率.在验证集上的accuracy.     
可以考虑绘制一下函数曲线图,可以更直观反映.稍后有时间补上.  

//TODO 网络准确度的衡量

通过在样本中随机取样吧，相同的和不同的数目相等, 具体逻辑可以看代码, totals代表总数, threshold是阈值.最后再绘制阈值不同时的函数变化曲线, 可以得到不同的识别率.        
不同的, 距离应该小于阈值，相同的则距离应该大于阈值.看一下下面的效果图. 很完美了, 大概取在0.82左右有0.88的准确度.    
每一次执行结果都有可能不同, 因为生成的验证集不一样，但是结果都不会差很多．
至于说这个函数画出来肯定是一个凸函数, 在(-1, 1)上取极大值...自变量threshold, 因变量_same_number和_diff_number, threshold变化时，肯定一个增大一个减小，在(-1, 1)之间肯定有一个焦点(不管他函数曲线是怎么样的).

```python
def generate_accuracy_map(features, labels, totals=6000, threshold=0):
    # the number of _diff and _same = totals/2
    _diff = []
    _same = []
    unique_labels = set(labels)
    length = len(unique_labels)
    diff_features = []
    for i in range(length):
        ith_features = features[labels==i]
        diff_features.append(ith_features)
        # 每个样本平均取
        for j in range(totals/(2*length)):
            x = random.randint(0, len(ith_features)-1)
            y = random.randint(0, len(ith_features)-1)
            first = ith_features[x]
            second = ith_features[y]
            # 这是所有相同的
            _same.append(cosine_distnace(first, second))
    # 这是不相同
    # 随机抽，不会抽在同一个类中
    for j in range(totals/2):
        while True:
            x = random.randint(0, length-1)
            y = random.randint(0, length-1)
            if x != y:
                break
        first = random.randint(0, len(diff_features[x])-1)
        second = random.randint(0, len(diff_features[y])-1)
        _diff.append(cosine_distnace(diff_features[x][first], diff_features[y][second]))
    correct = 0
    for elememt in _diff:
        if elememt >= threadhold:
            correct = correct + 1
    for elememt in _same:
        if elememt >= threadhold:
            correct = correct + 1
    return float(correct)/totals
```
![](http://omoitwcai.bkt.clouddn.com/2017-08-19-Figure_1.png)