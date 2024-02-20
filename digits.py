import numpy as np # linear algebra
import struct
import gzip
from array import array
from os.path  import join
import matplotlib.pyplot as plt


def main():
    print("Loading data...")
    input_path = './data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte.gz')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte.gz')
    (images, labels, label_digits) = load_data(training_images_filepath, training_labels_filepath)

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
    print("Training the model...")
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
        index = int(input("Enter a number (0 - 59999), or -1 for quit: "))
        if index < 0:
            return
        img = images[index]
        plt.figure(figsize=(10,7))
        plt.imshow(img.reshape(28, 28), cmap="Greys")
        plt.rcParams.update({'font.size': 25})


        img.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        plt.title(f"training image [{index}], label: {label_digits[index]}, recognized: {o.argmax()}")
        plt.show()


def load_data(images_filepath, labels_filepath):        
    labels = []
    with gzip.open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels_data = array("B", file.read())
        labels = np.eye(10)[labels_data]      
    
    with gzip.open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())
        images = np.reshape(image_data, (size, rows * cols))  
        images = images / 255    

    return images, labels, labels_data

if __name__ == "__main__":
    main()
