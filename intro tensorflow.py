import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

observations = 1000
xs = np.random.uniform(-10,10,(observations,1))
zs = np.random.uniform(-10,10,(observations,1)) 
generated_inputs = np.column_stack((xs,zs))
noise = np.random.uniform(-1,1,(observations,1))
generated_target = 2*xs - 3*zs + 5 + noise

np.savez('TF_intro', inputs = generated_inputs, targets=  generated_target)

training_data = np.load('TF_intro.npz')
input_size = 2
output_size = 1
model = tf.keras.Sequential([tf.keras.layers.Dense(output_size,
                            kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                            bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))]
                            )

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

#L2 Norm Loss lebih sensitif terhadap outliers, gunakan Huber loss jika terdapatoutliers karena Huber loss insensitive terhadap outliers
model.compile(optimizer=custom_optimizer,loss='mean_squared_error')
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=2)

#Layer hanya ada 1 sehingga ditulis 0
print(model.layers[0].get_weights())
weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print(weights,bias)

#membandingkan output dan target
print(model.predict_on_batch(training_data['inputs']).round(1))
print(training_data['targets'].round(1))

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()