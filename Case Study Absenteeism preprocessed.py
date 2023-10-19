import pandas as pd 

raw_data_csv = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/DEEP LEARNING/Absenteeism data.csv')

#print(type(raw_data_csv))
#print(raw_data_csv)

df = raw_data_csv.copy()

#MENGATUR DISPLAY
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

#print(df)
print(df.info())

#DROP ID
df = df.drop(['ID'], axis = 1)
print(df)
#print(raw_data_csv)


#MEMBUAT VARIABEL 'REASON FOR ABSENCE'
print(df['Reason for Absence'].min()) #nilai minimal 0
print(df['Reason for Absence'].max()) #nilai maksimal 28

print(pd.unique(df['Reason for Absence'])) #cara 1 lihat daftar data yang unik
print(df['Reason for Absence'].unique()) #cara 2

print(len(df['Reason for Absence'].unique()))
print(sorted(df['Reason for Absence'].unique())) #nilai 20 tdk ada datanya

#MEMBUAT DUMMY VARIABEL 'REASON TO ABSENCE'
reason_columns = pd.get_dummies(df['Reason for Absence'])
print(reason_columns)

#satu orang yang absen hanya bisa memiliki 1 alasan, tidak boleh lebih dari 1
reason_columns['check'] = reason_columns.sum(axis=1) #untuk cek reason hanya boleh 1
print(reason_columns) #cara 1 jika jumlah 'check' per row lebih 1 maka reason lebih dari 1

print(reason_columns['check'].sum(axis=0)) #jika jumlahnya sama dgn total data maka semua org absen dgn 1 alasan 
print(reason_columns['check'].unique()) #jika uniknya hanya 1 maka semua data sudah sesuai

#kemudian drop kolom "check" karena tidak digunakan lagi
reason_columns = reason_columns.drop(['check'], axis=1)
print(reason_columns)

#drop kolom pertama dalam feature dummy "reason columns"
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first= True)
print(reason_columns)

#Drop feature "reason for absence" yang bukan dummy di df karena feature ini akan diganti dengan variabel "reason columns"
df = df.drop(['Reason for Absence'], axis=1)
print(df)

#GROUP FEATURE "REASON FOR ABSENCE"
#karena dummy variabel "reason for absence" terlalu banyak, maka sebaiknya kita mengelompokan beberapa reason menjadi 1 reason
#reason 1-14 tentang berbagai penyakit sehingga bisa dibuat 1 variabel/group
#reason 15-17 tentang hamil dan melahirkan bisa 1 group
#reason 18-21 tentang poison or sign not elsewere categorize bisa 1 group
#reason 22-28 tentang light reason as konsultasi, dental, check up stc dan bisa 1 group

print(df.columns.values) #melihat values dari sebuah variabel df
print(reason_columns.columns.values) #melihat values dari sebuah feature

reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
print(reason_type_4)

#CONCAT ANTARA DF DAN REASON COLUMNS
df = pd.concat([df, reason_type_1,reason_type_2,reason_type_3,reason_type_4], axis=1)
print(df)

#ganti nama kolom
print(df.columns.values)
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets',
            'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns = column_names
print(df.head())

#reorder kolom atau mengubah urutan kolom
column_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets',
            'Absenteeism Time in Hours']

df = df[column_names_reordered]
print(df)

#CREATE CHECKPOINT 1
df_reason_mod = df.copy()
print(df_reason_mod.head())

#MENGELOLA FEATURE "DATE"
print(type(df_reason_mod['Date'])) #cek jenis data
print(type(df_reason_mod['Date'][0]))#cek jenis data di dalam data

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format='%d/%m/%Y')
print(df_reason_mod['Date'])
print(type(df_reason_mod['Date'])) #cek jenis data
print(type(df_reason_mod['Date'][0]))#cek jenis data di dalam data
print(df_reason_mod.info())

#EXTRACT THE MONTH VALUES
list_months = []

for  i in range(df_reason_mod.shape[0]): #shape adalah jumlah row dan kolom
    list_months.append(df_reason_mod['Date'][i].month) #extract bulan dari feature "Date"

print(len(list_months)) #cek jumlah dari feature baru sama dgn jumlah row di dataframe

df_reason_mod['Month Value'] = list_months #membuat variabul bulan di dataframe
print(df_reason_mod.head(20))

#EXTRACT THE DAY OF THE WEEK
def date_to_weekday(date_value):
    return date_value.weekday() #mengambil data hari dari variabel "Date"

df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
print(df_reason_mod.head())

#Drop feature "Date" karena sudah tidak terpakai
df_reason_mod = df_reason_mod.drop(['Date'], axis=1)
print(df_reason_mod.head())

#Reorder kolom
print(df_reason_mod.columns.values)
column_names_reordered_date = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
            'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index',
            'Education', 'Children', 'Pets', 'Absenteeism Time in Hours' ]

df_reason_mod = df_reason_mod[column_names_reordered_date]
print(df_reason_mod.head())

#CREATE CHECKPOINT 2
df_reason_date_mod = df_reason_mod.copy()
print(df_reason_date_mod.head())

#CEK VARIABEL biaya transport, jarak rumah, umur, beban kerja, dan index masa tubuh
print(type(df_reason_date_mod['Transportation Expense'][0]))
print(type(df_reason_date_mod['Distance to Work'][0]))
print(type(df_reason_date_mod['Age'][0]))
print(type(df_reason_date_mod['Daily Work Load Average'][0]))
print(type(df_reason_date_mod['Body Mass Index'][0]))

#MEMBUAT VARIABEL DUMMY EDUCATION
print(df_reason_date_mod['Education'].unique()) #melihat kategori yang ada di variabel education
print(df_reason_date_mod['Education'].value_counts()) #menghitung total kategori yang ada di variabel education 

df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
print(df_reason_date_mod['Education'].unique()) #melihat kategori yang ada di variabel education
print(df_reason_date_mod['Education'].value_counts()) #menghitung total kategori yang ada di variabel education 

#FINAL CHECKPOINT
df_preprocessed = df_reason_date_mod.copy()
print(df_preprocessed.head(10))

#Export to CSV
df_preprocessed.to_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/DEEP LEARNING/Absenteeism_data_preprocessed.csv', index=False)