import pandas as pd


def runProject():
    print(f"Reading Training Data Final")
    df = pd.read_table('train_data_final.csv')
    print(df.head(5))


if __name__ == "__main__":
    runProject()
