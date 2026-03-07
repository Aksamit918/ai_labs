import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def print_col_info(df):
    name = input("Column name: ").strip()

    if name in df.columns:
        print(df[name].describe())
    else:
        print(f"Error: Column '{name}' not found.")

def impute_null_values(df):
    df = df.dropna(subset=['Name', 'Cabin'])

    df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
    df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['VIP'] = df['VIP'].fillna(df['VIP'].mode()[0]).astype(bool)

    df['RoomService'] = df['RoomService'].fillna(0)
    df['FoodCourt'] = df['FoodCourt'].fillna(0)
    df['ShoppingMall'] = df['ShoppingMall'].fillna(0)
    df['Spa'] = df['Spa'].fillna(0)
    df['VRDeck'] = df['VRDeck'].fillna(0)

    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df.drop('Cabin', axis=1, inplace=True)

    print("Imputation completed successfully!")
    return df

def normalize_data(df):
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Normalization completed successfully!")
    return df

def convert_categorial(df):
    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Deck', 'Side', 'CryoSleep', 'VIP'], drop_first=True)
    print("Categorical encoding completed!")
    return df

def print_menu():
    print("\n--- Data Preprocessing Menu ---")
    print("1. Show Dataset Info")
    print("2. Show Missing Values")
    print("3. Detailed Column Information")
    print("4. Check Dataset Shape")
    print("5. Impute Missing Values")
    print("6. Convert Categorical to Numerical")
    print("7. Split Data into Train/Test (70/30)")
    print("8. Normalize Numeric Data")
    print("0. Exit")
    print("--------------------------------")

def main():
    df = pd.read_csv("train.csv")
    train_df = None
    test_df = None

    while True:
        print_menu()

        ch = input("Action: ").strip()
        try:
            ch = int(ch)
        except ValueError:
            break

        match ch:
            case 1:
                print(df.info())
            case 2:
                print(df.isnull().sum())
            case 3:
                print_col_info(df)
            case 4:
                print(df.shape)
            case 5:
                df = impute_null_values(df)
            case 6:
                df = convert_categorial(df)
            case 7:
                train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
                print(f"Split completed! Train shape: {train_df.shape}, Test shape: {test_df.shape}")
            case 8:
                if train_df is not None and test_df is not None:
                    train_df = normalize_data(train_df)
                    test_df = normalize_data(test_df)
                else:
                    print("Error: You must split the data before normalizing!")
            case 0:
                print("Loop has exited")
                break
            case _:
                print("End")

main()
