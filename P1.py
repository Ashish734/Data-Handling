import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"
df=pd.read_csv(url)
print(df.shape)
print(df.dtypes)
print(df.info())
print(df.describe())

