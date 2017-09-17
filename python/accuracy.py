import predict
import distance
import matplotlib.pyplot as plt
import pylab
import numpy as np

def generate_accuracy_with_threshold(source, caffemodel, deploy_file, threshold=0, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    features, samples, patch = predict.ordinary_predict_dataset(source=source, caffemodel=caffemodel, deploy_file=deploy_file, dimension=dimension, IMAGE_SIZE=IMAGE_SIZE, gpu_mode=gpu_mode, LAST_LAYER_NAME=LAST_LAYER_NAME)
    # for index, feature in enumerate(features):
    correct = 0
    totals = len(patch[0]) + len(patch[1])
    print patch[0]
    print patch[1]
    for x in patch[0]:
        s1, s2 = x
        i1 = samples.index(s1)
        i2 = samples.index(s2)
        print "i1, i2:%s %s" % (i1, i2)
        d = distance.cosine_distnace(features[i1], features[i2])
        print d
        if d >= threshold:
            correct = correct + 1
    print "------"
    for x in patch[1]:
        s1, s2 = x
        i1 = samples.index(s1)
        i2 = samples.index(s2)
        print "i1, i2:%s %s" % (i1, i2)
        d = distance.cosine_distnace(features[i1], features[i2])
        if d < threshold:
            correct = correct + 1
        print d
    return float(correct)/totals

def plot_accuracy_map(source, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    features, samples, patch = predict.ordinary_predict_dataset(source=source, caffemodel=caffemodel, deploy_file=deploy_file, dimension=dimension, IMAGE_SIZE=IMAGE_SIZE, gpu_mode=gpu_mode, LAST_LAYER_NAME=LAST_LAYER_NAME)
    # for index, feature in enumerate(features):
    totals = len(patch[0]) + len(patch[1])
    print patch[0]
    print patch[1]
    x_values = pylab.arange(-1.0, 1.01, 0.001)
    y_values = []
    for threshold in x_values:
        correct = 0
        print "------"*2
        for x in patch[0]:
            s1, s2 = x
            i1 = samples.index(s1)
            i2 = samples.index(s2)
            d = distance.cosine_distnace(features[i1], features[i2])
            print d
            if d >= threshold:
                correct = correct + 1
        print "******"
        for x in patch[1]:
            s1, s2 = x
            i1 = samples.index(s1)
            i2 = samples.index(s2)
            d = distance.cosine_distnace(features[i1], features[i2])
            print d
            if d < threshold:
                correct = correct + 1
        print "------"*2
        y_values.append(float(correct)/totals)
    max_index = np.argmax(y_values)
    print max_index
    plt.title("threshold-accuracy curve")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_values, y_values)
    plt.plot(x_values[max_index], y_values[max_index], '*', color='red', label="(%s, %s)"%(x_values[max_index], y_values[max_index]))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_accuracy_map("/Users/HZzone/Desktop/temp/test", "/Users/HZzone/Downloads/lenet_siamese_train_iter_15000.caffemodel", "../ct-test/lenet_siamese.prototxt", gpu_mode=False, LAST_LAYER_NAME="feat")
    # print generate_accuracy_with_threshold("/Users/HZzone/Desktop/temp", "/Users/HZzone/Downloads/lenet_iter_10000.caffemodel", "../ct-test/lenet.prototxt", gpu_mode=False, threshold=0.97)

