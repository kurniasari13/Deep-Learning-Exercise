import numpy as np 
import pandas as pd

### Load Data ###
data_preprocessed = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/DEEP LEARNING/Absenteeism preprocessed.csv')
print(data_preprocessed.head())

### Mengelola Target ###
print(data_preprocessed['Absenteeism Time in Hours'].median())

targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                    data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)

print(targets)

### Masukan target ke data table ###
data_preprocessed['Excessive Absenteeism'] = targets
print(data_preprocessed.head())

### Balance Dataset ###
print(targets.sum() / targets.shape[0]) # logistik kalo 45% masih oke, kalo neural network harus 50%

### Drop variabel "Absenteeism time in hurs" ###
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 'Day of the Week',
                                            'Daily Work Load Average', 'Distance to Work'], axis=1) #3 variabel di akhir di drop karena weightnya mendekati no, yang artinya tidak signifikan

print(data_with_targets is data_preprocessed) #cek apakah data tabel baru masih sama dengan sata tabel lama
print(data_with_targets.head())

### Select Inputs ###
data_with_targets.iloc[:,:14] #cara 1
data_with_targets.iloc[:,:-1] #Cara 2

unscaled_inputs = data_with_targets.iloc[:,:-1] #Cara 2
print(unscaled_inputs.head())

### Scaled Inputs ###
#CARA 1 PAKE STANDARD SCALER
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
#Absenteeism_scaler = StandardScaler()

#CARA 2 PAKE CUSTOM SCALER: DUMMY TIDAK DI SCALE
class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

#CARA 1 MEMBUAT LIST NAMA KOLOM
#print(unscaled_inputs.columns.values)
#columns_to_scale = ['Month Value', 'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
                    #'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pet'] 

#CARA 2 MEMBUAT LIST KOLOM NAMA
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]

absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
print(scaled_inputs)
print(scaled_inputs.shape)

### Split and Shuffle Inputs ###
from sklearn.model_selection import train_test_split #di sklearn split sudah termasuk shuffle

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

### Logistik Regression ###
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg = LogisticRegression()
reg.fit(x_train, y_train)
print(reg.score(x_train, y_train)) #cek akurasi

### Manually Cek Accuracy ###
model_outputs = reg.predict(x_train)
print(model_outputs) #membandingkan output model dengan target
print(y_train)

print(model_outputs == y_train) #compare output model dengan target
print(np.sum((model_outputs == y_train))) #menghitung jumlah output model yang sama dengan target
print(model_outputs.shape[0]) #total nuber of elements

#akurasi: total True dibagi dengan total element dari model output
print(np.sum((model_outputs == y_train)) / model_outputs.shape[0])

### Finding Intercept and Coefficient ###
print(reg.intercept_)
print(reg.coef_)

print(unscaled_inputs.columns.values)
feature_names = unscaled_inputs.columns.values

summary_table = pd.DataFrame(columns=['Feature Name'], data= feature_names)
summary_table['Coefficient'] = np.transpose(reg.coef_)
print(summary_table)

### Adding Intercept in Summary Table ###
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
print(summary_table)

### Interpreting the Coefficient ###
summary_table['Odds_Ratio'] = np.exp(summary_table.Coefficient)
print(summary_table.sort_values('Odds_Ratio', ascending=False))
# koefisine mendekati 0 atau odds ratio mendekati 1 maka pengaruhnya tidak signifikan

### Backward Elimination ###
#variabel yang tidak signifikan di drop (weight mendekati 0 atau odds ratio mendekati 1)
#drop variabel dengan cara kembali ke checkpoint sebelum scale input dan drop variabel  yang tidak signifikan

### Tets the Model ###
print(reg.score(x_test, y_test))
#mencari probabilias for 1 to be excessive absent
predicted_probably = reg.predict_proba(x_test)
print(predicted_probably) # ada 2 kolom, sebelah kiri proba being 0, sebelah kanan proba being 1
print(predicted_probably.shape)
print(predicted_probably[:, 1])

### Save the Model ###
import pickle
with open('model', 'wb') as file:
    pickle.dump(reg, file)

with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)

