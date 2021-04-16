import tensorflow as tf
from tensorflow.keras import datasets,layers,models,losses
import matplotlib.pyplot as plt

#import fashion mnist data from keras
fashion_mnist_data = tf.keras.datasets.fashion_mnist
(tr_images, tr_labels), (test_images, test_labels) = fashion_mnist_data.load_data()


print(tr_labels)


class_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
print(tr_images.shape)
print(tr_labels.shape)

#view data images
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.subplots_adjust(hspace=.3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(tr_images[i])
    plt.colorbar()
    plt.grid(True)
    plt.title(class_names[tr_labels[i]])
plt.show()


#Begin CNN Model
tr_images = tr_images/255.0
test_images = test_images/255.0
X_train = tr_images.reshape((tr_images.shape[0],28,28,1))
X_test = test_images.reshape((test_images.shape[0],28,28,1))

tf.random.set_seed(42)
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss = losses.sparse_categorical_crossentropy,metrics=['accuracy'])
model.summary()

model.fit(X_train,tr_labels,validation_data = (X_test, test_labels), epochs=1)

predictions = model.predict(X_test)


