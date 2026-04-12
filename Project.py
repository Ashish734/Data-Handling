import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Indian Air Quality Trends - 2023-2024.csv")
print(df.shape)
print(df.describe())
print(df.info())
print(df.columns)
print("Missing Values:\n",df.isnull().sum())
print("Duplicate Values:",df.duplicated().sum())
def AQI_Category(pm25):
    if pm25 <= 30:
        return 'Good'
    elif pm25 <=60:
        return 'Satisfactory'
    elif pm25 <=90:
        return 'Moderate'
    elif pm25 <=120:
        return 'Poor'
    elif pm25 <=250:
        return 'Very Poor'
    else:
        return 'Severe'
df['AQI_Category']=df['pm2_5'].apply(AQI_Category)
print(df.columns)
print(f"Cities: {df['city'].unique()}")
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='city', y='pm2_5')
plt.title('PM2.5 Distribution Across Cities', fontsize=14, fontweight='bold')
plt.xlabel('City')
plt.ylabel('PM2.5 (µg/m³)')
plt.show()
