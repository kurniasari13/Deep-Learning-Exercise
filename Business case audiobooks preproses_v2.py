import numpy as np 
from sklearn import preprocessing

raw_csv_data = np.loadtxt('D:/DATA ANALYST/belajar_python/LATIHAN DS/DEEP LEARNING/Audiobooks data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]

#SHUFFLE DATA SEBELUM BALANCE DATASET
shuffled_indices = np.arange(unscaled_inputs_all[0])
np.random.shuffle(shuffled_indices)

unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
targets_all = targets_all[shuffled_indices]

#BALANCE THE DATASET
num_one_targets = int(np.sum(targets_all)) #menghitung target yang 1
#print(num_one_targets)
zero_targets_counter = 0  #menghitung target yang 0
indices_to_remove = [] #list row yang perlu dihapus

for i in range(targets_all.shape[0]): #shape[0] sama dengan total jumlah row yang ada di dataset
    if targets_all[i] ==0:  
        zero_targets_counter += 1 #saat target sama dengan nol maka zero_target_counter bertambah 1 dan seterusnya (dengan kata lain sedang menghitung jumlah target yang nol)
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i) #saat jumlah target 1 sudah sama dengan jumlah target nol, then iterate selanjutnya python akan memasukan list row ke dalam variabel indices_to_remove untuk dihapus

#print(zero_targets_counter)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

#membandingkan sebelum dan sesudah di balance dataset
#print(len(unscaled_inputs_all))
#print(len(targets_all))

#print(len(unscaled_inputs_equal_priors))
#print(len(targets_equal_priors))

#SCALED INPUT
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)


#SHUFFLE DATASET KARENA NANTI AKAN MENGGUNAKAN BATCHING METHOD
shuffled_indices = np.arange(scaled_inputs.shape[0]) #membuat list dari order of data
print(shuffled_indices) 

np.random.shuffle(shuffled_indices) #setelah mendapatlan list order, lalu shuffle dan masukan ke dalam data
print(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

#Bandingan data sebelum dan sesudah shuffle
#print(scaled_inputs)
#print(shuffled_inputs)

#SPLIT DATASET
samples_count = shuffled_inputs.shape[0]
print(samples_count)

train_samples_count = int(0.8 * samples_count) #buat variabel jumlah tiap subdata
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

#Melihat balance dari train, validation, dan test dataset
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

#Simpan tiga dataset ke .npz file
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)

