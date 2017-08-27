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
    for x in patch[0]:
        s1, s2 = x
        i1 = samples.index(s1)
        i2 = samples.index(s1)
        d = distance.cosine_distnace(features[i1], features[i2])
        if d >= threshold:
            correct = correct + 1
    for x in patch[1]:
        s1, s2 = x
        i1 = samples.index(s1)
        i2 = samples.index(s1)
        d = distance.cosine_distnace(features[i1], features[i2])
        if d < threshold:
            correct = correct + 1
    return float(correct)/totals

def plot_accuracy_map(source, caffemodel, deploy_file, dimension=150, IMAGE_SIZE=227, gpu_mode=True, LAST_LAYER_NAME="ip1"):
    x_vaules = pylab.arange(-1.0, 1.01, 0.01)
    y_values = []
    for x in x_vaules:
        y_values.append(generate_accuracy_with_threshold(source=source, caffemodel=caffemodel,deploy_file=deploy_file, dimension=dimension, IMAGE_SIZE=IMAGE_SIZE, gpu_mode=gpu_mode, LAST_LAYER_NAME=LAST_LAYER_NAME, threshold=x))
    max_index = np.argmax(y_values)
    print max_index
    plt.title("threshold-accuracy curve")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_vaules, y_values)
    plt.plot(x_vaules[max_index], y_values[max_index], '.', label="(%s, %s)"%(x_vaules[max_index], y_values[max_index]))
    plt.title("Threshold-Accuracy")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_vaules, y_values)
    plt.plot(x_vaules[max_index], y_values[max_index], '*', color='red', label="(%s, %s)"%(x_vaules[max_index], y_values[max_index]))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_accuracy_map("/Users/HZzone/Desktop/temp", "/Users/HZzone/Downloads/lenet_iter_10000.caffemodel", "../ct-test/lenet.prototxt", gpu_mode=False)

