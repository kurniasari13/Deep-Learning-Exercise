import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

#BIKIN VARIABEL INDEPENDEN
observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations,1))
zs = np.random.uniform(-10,10,(observations,1)) 
inputs = np.column_stack((xs,zs))
print(inputs.shape)

noise = np.random.uniform(-1,1,(observations,1))
targets = 2*xs - 3*zs + 5 + noise 
print(targets.shape)

#GRAFIK
targets = targets.reshape(observations,) 
print(targets.shape)
xs = xs.reshape(observations,) 
zs = zs.reshape(observations,) 
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xs, zs, targets)
#ax.plot(xs, zs, targets)
#ax.set_xlabel('xs')
#ax.set_ylabel('zs')
#ax.set_zlabel('Targets')
#ax.view_init(azim=100)
#plt.show()

#RESHAPE LAGI SETELAH PLOTTING
targets = targets.reshape(observations,1)
xs = xs.reshape(observations,1) 
zs = zs.reshape(observations,1) 
print(targets.shape)
print(inputs.shape)

#INITIALIZE WEIGHT AND BIAS
init_range = 0.1
weights = np.random.uniform(low=-init_range, high=init_range,size=(2,1))
biases = np.random.uniform(low=-init_range,high=init_range,size=1)
print(weights)
print(biases)

#SET A LEARNING RATE
learning_rate = 0.02

for i in range (100):
    outputs = np.dot(inputs,weights) + biases 
    deltas = outputs - targets
    loss = np.sum(deltas**2) / 2 / observations 
    print(loss) 
    deltas_scaled = deltas / observations 
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

print(weights,biases)

#plt.plot(outputs, targets)
#plt.xlabel('Outputs')
#plt.ylabel('Targets')
#plt.show()
