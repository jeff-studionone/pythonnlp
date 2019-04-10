import tensorflow as tf
import numpy as np
import keras

# Earlier when you trained for extra epochs you had an issue where your loss might change. It might have taken a bit of time for you to wait for the training to do that, and you might have thought 'wouldn't it be nice if I could stop the training when I reach a desired value?' -- i.e. 95% accuracy might be enough for you, and if you reach that after 3 epochs, why sit around waiting for it to finish a lot more epochs....So how would you fix that? Like any other program...you have callbacks! Let's see them in action...

print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True


# fashion dataset
mnist = tf.keras.datasets.fashion_mnist
# get data training and test images
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalizing data
training_images=training_images/255.0
test_images=test_images/255.0

# callback
callbacks = myCallback()

# creating model, flatten data, dense layer and final layer
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape(28,28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# define loss function
model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
     )
# training model
print('Training data ====> ')
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

print('Test data ====> ')
model.evaluate(test_images, test_labels)

# made a prediction
classifications = model.predict(test_images)

print('Get the classification value ===>')
print(classifications[0])
print(test_labels[0])
