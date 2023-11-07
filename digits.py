import numpy as np # linear algebra
import struct
import gzip
from array import array
from os.path  import join
import random
import matplotlib.pyplot as plt


#
# MNIST Data Loader and Plotter 
#
class MnistData(object):
    def load_data(self, images_filepath, labels_filepath):        
        labels = []
        with gzip.open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels_data = array("B", file.read())
            labels = np.eye(10)[labels_data]      
        
        with gzip.open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            print(str((magic, size, rows, cols)))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
            #image_data = image_data.astype("float32") / 255
            image_data = image_data / 255
            images = np.reshape(image_data, (size, rows * cols))   

        print(str(labels.shape))
        print(str(images.shape))
        print(str(labels[10]))
        print(str(images[10]))
        
        
        return images, labels
            
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
    
    def show_image(self, image, title_text):
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title(title_text, fontsize = 14)
        plt.show()
            
 



def main():
    print("Loading data...")
    input_path = './data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte.gz')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte.gz')
    mnist_data = MnistData()
    (images, labels) = mnist_data.load_data(training_images_filepath, training_labels_filepath)


    print("Training the model...")

    """
    w = weights, b = bias, i = input, h = hidden, o = output, l = label
    e.g. w_i_h = weights from input layer to hidden layer
    """
    w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
    w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
    b_i_h = np.zeros((20, 1))
    b_h_o = np.zeros((10, 1))

    learn_rate = 0.01
    nr_correct = 0
    epochs = 3
    for epoch in range(epochs):
        for img, l in zip(images, labels):
            img.shape += (1,)
            l.shape += (1,)
            # Forward propagation input -> hidden
            h_pre = b_i_h + w_i_h @ img
            h = 1 / (1 + np.exp(-h_pre))
            # Forward propagation hidden -> output
            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            # Cost / Error calculation
            e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
            nr_correct += int(np.argmax(o) == np.argmax(l))

            # Backpropagation output -> hidden (cost function derivative)
            delta_o = o - l
            w_h_o += -learn_rate * delta_o @ np.transpose(h)
            b_h_o += -learn_rate * delta_o
            # Backpropagation hidden -> input (activation function derivative)
            delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
            w_i_h += -learn_rate * delta_h @ np.transpose(img)
            b_i_h += -learn_rate * delta_h

        # Show accuracy for this epoch
        print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0

    # Show results
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        img.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        plt.title(f"training image [{index}], label: {labels[index]}, recognized: {o.argmax()}")
        plt.show()


    r = random.randint(1, 60000)
    mnist_data.show_image(images[r], 'training image [' + str(r) + '] = ' + str(labels[r]))

    r = random.randint(1, 60000)
    mnist_data.show_image(images[r], 'training image [' + str(r) + '] = ' + str(labels[r]))







if __name__ == "__main__":
    main()



