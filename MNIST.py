import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds 
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True) 

#mengambil data raw
mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)

#data raw di scale atau fiture scaling, tanda titik artinya float, /= artinya dibagi dengan
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)

#scale dan membuat variabel data test
test_data = mnist_test.map(scale)

# buffer data atau shuffle datanya diacak agar random, shuffle diperlukan jika ingin menggunakan batch
BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

#membuat variabel data validation dan data train yang diambil dari data yang sudah di suffle
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

#membuat batch agar mempermudah algoritma optimization bekerja, loss disini hanya loss rata-rata dari batch
BATCH_SIZE = 100
train_data = train_data.batch(BATCH_SIZE)

#validation dan test tidak perlu batch karena hanya forward propagation dan tdk membutuhkan komputasi besar
#validation dan test juga perlu exact values loss, sehingga kalo pakai batch hanya get average loss bukan exact, so better not use batch
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

#validation data harus punya same shape object properties seperti train dan test data, MNIST data is iterable dan 2-tuple format (as_supervised=TRUE)
#we must extract and convert validation input dan target properly
#iter adalah membuat objek yang mana dapat diitered dalam satu waktu (forloop dan whileloop), tapi tidak load data
# next is loads the next batch (loads the next element of an iterable object)
validation_inputs, validation_targets = next(iter(validation_data))

#OUTLINE THE MODEL
input_size = 784
output_size = 10
hidden_layer_size = 50

#pertama input layer, dilanjut layer kedua dan ketiga yaitu hidden layer dengan menulis jumlah output atau jumlah unit di hidden layer dan jenis activationnya
#dan yang terakhir adalah output layer dengan menulis jumlah output yang diinginkan dan jenis activationnya

model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                tf.keras.layers.Dense(output_size, activation='softmax')
                            ])

#jika output dalam bentuk binary, maka loss function yang tepat adalah cross tropy
#ada 3 jenis cross tropy, binary crosstropy, categoical crosstropy, dan sparse categorical crosstropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#jika ingin learning rate custom maka:
#custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#TRAIN MODEL: fit model dengan train data, total epoch, dan validation data
#The validation data we just created ourselves in the format: (inputs,targets)
NUM_EPOCHS = 5
model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs,validation_targets), verbose=2)

#TEST MODEL
test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


#semua percobaan akurasi sekitar 97%
# jika ingin akurasi diatas 98.5%, maka hidden layer size naik jd 5000, jumlah hiiden layer naik jd 10, batch size naik 150, number epoch naik 10
# dengan metode diatas, karena width dan depth yang besar butuh waktu 3 jam lebih untuk komputer running model