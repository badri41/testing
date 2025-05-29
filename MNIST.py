import numpy as np
import pandas as pd

#Lets Build my first ML Project for FUN

train_path = r'C:\Users\Dell\OneDrive\Desktop\office work\mnist_train_small.csv'
test_path = r'C:\Users\Dell\OneDrive\Desktop\office work\mnist_test.csv'

"""
    All Hyper Parameters:
    Number of Batches : 32 
    Learning Rate : 0.01
    filter size : 3 x 3
    MaxPooling size : 2 x 2
"""

class ParseData:
    """We Have to parse the data in such a way that it will be most easier for our CNN to learn and also divide it into batches 
    """
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.num_classes = 10 # 10 outputs 0 to 9
        
        self.X_train, self.Y_train = self.preprocess(self.train_data)
        self.X_test, self.Y_test = self.preprocess(self.test_data) 
        
        self.train_batches = self.create_batches(self.X_train, self.Y_train, batch_size=32) #Divide the whole data into 32 batches to train

    def preprocess(self, data):
        labels = data.iloc[:, 0].values
        images = data.iloc[:, 1:].values / 255.0 #Normalized it as it is from 0 to 255
        images = images.reshape((-1, 1, 28, 28)) # Need to convert it to a standard 28 x 28 image for CNN processing

        one_hot_labels = np.zeros((labels.size, self.num_classes)) #One Hot Labels only fireup (1) the correct value
        one_hot_labels[np.arange(labels.size), labels] = 1

        return images, one_hot_labels

    def create_batches(self, X, Y, batch_size): 
        batches = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            batches.append((X_batch, Y_batch))
        return batches


class Layer:
    def __init__(self, in_channels, out_channels, filter_size = 3):
        self.filters = np.random.randn(out_channels, in_channels, filter_size, filter_size) * np.sqrt(2 / (in_channels * filter_size * filter_size)) #He Initialisation
        self.biases = np.zeros(out_channels) #biases always 0

    def conv_forward(self, input):
        batch_size, _, h, w = input.shape
        num_filters, _, f_h, f_w = self.filters.shape
        out_h = h - f_h + 1
        out_w = w - f_w + 1
        output = np.zeros((batch_size, num_filters, out_h, out_w))

        for b in range(batch_size): # b refers to each image in the 32 batch size
            for f in range(num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        region = input[b, :, i:i+f_h, j:j+f_w]
                        output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        cache = (input, self.filters, self.biases)
        return output, cache

    def conv_backward(self, d_out, cache):
        """
        Backward pass through convolutional layer.

        Args:
            d_out: Gradient of loss w.r.t. output of conv layer, shape (batch_size, num_filters, out_h, out_w)
            cache: Tuple (input, filters, biases)

        Returns:
            d_input: Gradient w.r.t. input, shape same as input
            d_filters: Gradient w.r.t. filters, shape same as filters
            d_biases: Gradient w.r.t. biases, shape same as biases
        """
        input, filters, biases = cache
        batch_size, in_channels, h, w = input.shape
        num_filters, _, f_h, f_w = filters.shape
        _, _, out_h, out_w = d_out.shape

        d_input = np.zeros_like(input)
        d_filters = np.zeros_like(filters)
        d_biases = np.zeros_like(biases)

        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i
                        h_end = h_start + f_h
                        w_start = j
                        w_end = w_start + f_w

                        region = input[b, :, h_start:h_end, w_start:w_end]
                        
                        # Gradients
                        d_filters[f] += d_out[b, f, i, j] * region
                        d_input[b, :, h_start:h_end, w_start:w_end] += d_out[b, f, i, j] * filters[f]
                        d_biases[f] += d_out[b, f, i, j]
    #Ok This stuff, I have an Idea of exactly whats happening but implementing them without anything is tedious
    # and useless and boring :< 
        return d_input, d_filters, d_biases

    
    def relu(self, input):
        output = np.maximum(0, input)
        cache = input
        return output, cache
    
    def relu_backward(self, d_out, cache):
        x = cache
        d_input = d_out * (x > 0)  # Gradient flows only where input > 0 simple :)
        return d_input


    def max_pool(self, input, size=2, stride=2): # 2 x 2 Max Pooling
        batch_size, channels, h, w = input.shape
        out_h = (h - size) // stride + 1
        out_w = (w - size) // stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = input[b, c, i*stride:i*stride+size, j*stride:j*stride+size]
                        output[b, c, i, j] = np.max(region)
        cache = (input, size, stride)
        return output, cache
    
    def pool_backward(self, d_out, cache):
        """
        Backpropagation through Max Pooling layer.
        Only the max index in each pool region receives the gradient and rest all terms be 0 ......

        Args:
            d_out: Gradient of the loss w.r.t. output of maxpool, shape (batch_size, channels, h_out, w_out)
            cache: Tuple (input, pool_size, stride)

        Returns:
            d_input: Gradient of the loss w.r.t. input of maxpool, shape same as input
        """
        input, pool_size, stride = cache
        batch_size, channels, h_in, w_in = input.shape
        h_out, w_out = d_out.shape[2], d_out.shape[3]

        d_input = np.zeros_like(input)
     
        for i in range(batch_size):
            for c in range(channels):
                for h in range(h_out):
                    for w in range(w_out):
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size

                        region = input[i, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        mask = (region == max_val)

                        d_input[i, c, h_start:h_end, w_start:w_end] += d_out[i, c, h, w] * mask
        return d_input

    def update_params(self, d_filters, d_biases, lr=0.01):
        """
        Apply gradient descent update to filters and biases.
        """
        self.filters -= lr * d_filters
        self.biases -= lr * d_biases


class NeuralNet:
    """The Neural Network of my Model which contains all Layers and forward pass and backward propagation
    """
    def __init__(self):
        self.layer1 = Layer(1, 8)
        self.layer2 = Layer(8, 16)
        self.layer3 = Layer(16, 32)
        self.W_fc = np.random.randn(10, 32) * np.sqrt(2 / 32) #Weights and Biases for the final dense layer
        self.b_fc = np.zeros(10) #Noted that biases are always initialized to 0

    def flatten(self, input):
        cache = input.shape
        return input.reshape((input.shape[0], -1)), cache
    
    def flatten_backward(self, d_out, original_shape):
        return d_out.reshape(original_shape)

    def softmax(self, x): 
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def cross_entropy_with_logits(self, logits, labels):
        probs = self.softmax(logits)
        loss = -np.mean(np.sum(labels * np.log(probs + 1e-8), axis=1)) #
        d_out = (probs - labels) / logits.shape[0]  #derivative of out L(y, p) = -1/N*sum(sum(ylog(pk))). dL/dzk = pk - yk 
        return loss, d_out

    def fc_backward(self, d_out, cache):
        """
        Fully connected backward pass.
        Args:
            d_out: Gradient of loss w.r.t. logits (batch_size x 10)
            cache: (flatten_output, W_fc, b_fc)
        Returns:
            d_input: Gradient w.r.t. FC input (batch_size x 32)
            dW_fc: Gradient w.r.t. weights
            db_fc: Gradient w.r.t. biases
        """
        flatten_out, W_fc, b_fc = cache

        dW_fc = d_out.T @ flatten_out
        db_fc = np.sum(d_out, axis=0)
        d_input = d_out @ W_fc

        return d_input, dW_fc, db_fc
    
    def forward(self, X):
        """
        Forward propagation:
        Input → Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → Conv3 → ReLU3 → Pool3
        → Flatten → Fully Connected → Logits
        """

        # Layer 1
        out1, self.cache_conv1 = self.layer1.conv_forward(X)
        out1, self.cache_relu1 = self.layer1.relu(out1)
        out1, self.cache_pool1 = self.layer1.max_pool(out1)

        # Layer 2
        out2, self.cache_conv2 = self.layer2.conv_forward(out1)
        out2, self.cache_relu2 = self.layer2.relu(out2)
        out2, self.cache_pool2 = self.layer2.max_pool(out2)

        # Layer 3
        out3, self.cache_conv3 = self.layer3.conv_forward(out2)
        out3, self.cache_relu3 = self.layer3.relu(out3)
        out3, self.cache_pool3 = self.layer3.max_pool(out3)

        # Flatten
        flat_out, self.cache_flatten = self.flatten(out3)

        # Fully Connected (FC) Layer → just logits, no softmax here
        logits = flat_out @ self.W_fc.T + self.b_fc  # (batch_size x 10)
        self.cache_fc = (flat_out, self.W_fc, self.b_fc)  # save for fc backward

        return logits

    
    def backward(self, d_out):
        """
            BackPropagation implementation to carryout the SGD and calculate the gradients as we go backwards into the layer using the 
            chain rule, cache, and derivatives of the functions
            d_out means we are alrady backward the softmax and cross entropy step
            flow of control
            Loss + softmax -> fc_backward -> flatten_backward -> Layer3 -> Layer2 -> Layer1 -> Yeahhhhhh
        Args:
            d_out (ndarray): d_out which is the derivative of the output of the softmax + cross entropy with logits
        """
        # FC layer
        d_out, self.dW_fc, self.db_fc = self.fc_backward(d_out, self.cache_fc)
        self.W_fc -= 0.01 * self.dW_fc
        self.b_fc -= 0.01 * self.db_fc

        # Flatten
        d_out = self.flatten_backward(d_out, self.cache_flatten)

        # Pool3 → ReLU3 → Conv3
        d_out = self.layer3.pool_backward(d_out, self.cache_pool3)
        d_out = self.layer3.relu_backward(d_out, self.cache_relu3)
        d_out, self.dW3, self.db3 = self.layer3.conv_backward(d_out, self.cache_conv3)
        self.layer3.update_params(self.dW3, self.db3, lr=0.01)

        # Pool2 → ReLU2 → Conv2
        d_out = self.layer2.pool_backward(d_out, self.cache_pool2)
        d_out = self.layer2.relu_backward(d_out, self.cache_relu2)
        d_out, self.dW2, self.db2 = self.layer2.conv_backward(d_out, self.cache_conv2)
        self.layer2.update_params(self.dW2, self.db2, lr=0.01)

        # Pool1 → ReLU1 → Conv1
        d_out = self.layer1.pool_backward(d_out, self.cache_pool1)
        d_out = self.layer1.relu_backward(d_out, self.cache_relu1)
        d_out, self.dW1, self.db1 = self.layer1.conv_backward(d_out, self.cache_conv1)
        self.layer1.update_params(self.dW1, self.db1, lr=0.01)


class Model:
    """My ML Model mwwuahhh
    """
    def __init__(self, train_path, test_path):
        self.data = ParseData(train_path, test_path)  # Self.data is an instance of the class ParseData
        self.net = NeuralNet()

    def shuffle_training_data(self):
        # Shuffle training data and recreate batches
        indices = np.arange(len(self.data.X_train))
        np.random.shuffle(indices)
        self.data.X_train = self.data.X_train[indices]
        self.data.Y_train = self.data.Y_train[indices]
        self.data.train_batches = self.data.create_batches(self.data.X_train, self.data.Y_train, batch_size=32)

    def train(self, epochs=10):
        """Flow of training is first run over all epochs then for in every epoch for every batch do 
                forward prop -> cross categorical entropy loss, Back Propagation -> Update Weights and Biases with Stochastic Gradient Descent
        """
        print("▶️  Training begin YAYYYYY Lesss gooooooooooo")
        for epoch in range(epochs):
            self.shuffle_training_data()  # Shuffle at the start of each epoch
            print(f"-- Starting epoch {epoch+1}")
            total_loss = 0
            for X_batch, Y_batch in self.data.train_batches:
                logits = self.net.forward(X_batch)
                loss, d_out = self.net.cross_entropy_with_logits(logits, Y_batch)
                total_loss += loss
                self.net.backward(d_out)  # Back Propagation and update of each parameter

            avg_loss = total_loss / len(self.data.train_batches)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    def evaluate(self, batch_size=32):
        X_test, Y_test = self.data.X_test, self.data.Y_test
        num_samples = len(X_test)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        all_preds = []
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            X_batch = X_test[start:end]
            logits = self.net.forward(X_batch)
            preds = np.argmax(logits, axis=1)
            all_preds.append(preds)

        predicted_labels = np.concatenate(all_preds)
        true_labels = np.argmax(Y_test, axis=0) if Y_test.ndim > 1 else Y_test  # safeguard for shape
        true_labels = np.argmax(Y_test, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

 
Yayy = Model(train_path, test_path)
Yayy.train()
Yayy.evaluate()