import pandas as pd

df = pd.read_csv("train.csv")

#print(df.head())
#print(df.info())
#print(df.dtypes)

# cols = df.columns.tolist()
# for col in cols:
#     print(col)

# obj = 'Name'
# if obj in cols:
#     print('Dataset has column Name')
# obj = 'Drowned'
# if obj not in cols:
#     print('Dataset hasn’t column Drowned')

# for col in df:
#     print(col)

print(df.isnull().sum())



