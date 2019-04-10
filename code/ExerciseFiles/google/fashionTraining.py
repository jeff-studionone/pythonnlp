import tensorflow as tf
import numpy as np
import keras

mnist = tf.keras.datasets.fashion_mnist

# calling load_data on this object will give you two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# what does these values look like?
# Let's print a training image, and a training label to see...Experiment with different indices in the array.
# For example, also take a look at index 42...that's a a different boot than the one at index 0

# import matplotlib.pyplot as plt
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

# You'll notice that all of the values in the number are between 0 and 255.
#  If we are training a neural network, for various reasons it's easier if we treat all values as between 0 and 1, a process called 'normalizing
# fortunately in Python it's easy to normalize a list like this without looping. You do it like this:

training_images = training_images / 255.0
test_images = test_images / 255.0

# Now you might be wondering why there are 2 sets...training and testing
# -- remember we spoke about this in the intro? The idea is to have 1 set of data for training, and then another set of data
# ...that the model hasn't yet seen...to see how good it would be at classifying values.
# After all, when you're done, you're going to want to try it out with data that it hadn't previously seen!

# Let's now design the model. There's quite a few new concepts here, but don't worry, you'll get the hang of them.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# SEQUENTIAL: That defines a SEQUENCE of layers in the neural network
# FLATTEN: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.
# DENSE: Adds a layer of neurons
# Each layer of neurons need an ACTIVATION function to tell them what to do. There's lots of options, but just use these for now.
# RELU effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.
# SOFTMAX takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

# The next thing to do, now the model is defined, is to actually build it. You do this by compiling it with an optimizer and loss function as before -- and then you train it by calling model.fit asking it to fit your training data to your training labels -- i.e. have it figure out the relationship between the training data and its actual labels, so in future if you have data that looks like the training data, then it can make a prediction for what that data would look like.

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Training data ====> ')
model.fit(training_images, training_labels, epochs=30)

# Once it's done training -- you should see an accuracy value at the end of the final epoch. It might look something like 0.9098. This tells you that your neural network is about 91% accurate in classifying the training data. I.E., it figured out a pattern match between the image and the labels that worked 91% of the time. Not great, but not bad considering it was only trained for 5 epochs and done quite quickly.
#
# But how would it work with unseen data? That's why we have the test images. We can call model.evaluate, and pass in the two sets, and it will report back the loss for each. Let's give it a try:

print('Test data ====> ')
model.evaluate(test_images, test_labels)

# made a prediction
classifications = model.predict(test_images)

# Let's now look at the layers in your model. Experiment with different values for the dense layer with 512 neurons. What different results do you get for loss, training time etc? Why do you think that's the case?
# ANSWER: Training takes longer, but is more accurate

print('Get the classification value ===>')
print(classifications[0])
print(test_labels[0])
