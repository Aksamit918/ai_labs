import pandas as pd

df = pd.read_csv('train.csv')

print(df.info())
print(df.isnull().sum())

cols_to_drop = ['PassengerId', 'Cabin', 'Name', 'VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt', 'RoomService', 'VIP']
df = df.drop(columns=cols_to_drop)

df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0]).astype(bool)
df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
df['Age'] = df['Age'].fillna(df['Age'].median())

df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)
df = pd.get_dummies(df, columns=['Destination'], drop_first=True)

print(df.info())
print(df.isnull().sum())
