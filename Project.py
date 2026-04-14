import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Indian Air Quality Trends - 2023-2024.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.columns)
print("Missing Values:\n",df.isnull().sum())
print("Duplicate Values:",df.duplicated().sum())
print(df['city'].unique())
df['date']=pd.to_datetime(df['date'])
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
print(df.describe())
for city, group in df.groupby('city'):
    plt.plot(group['date'],group['pm2_5'])
#plt.plot(df['date'],df['pm2_5'])
plt.legend(["Delhi","Mumbai","Bengaluru","Chennai","Kolkata","Hyderabad"],loc="upper right")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("PM2.5 Over Time")
plt.show()
city_avg = df.groupby('city').mean(numeric_only=True)
pollutants = {
    'pm2_5': 'PM2.5',
    'pm10': 'PM10',
    'carbon_monoxide': 'CO',
    'nitrogen_dioxide': 'NO2'
}
for col, label in pollutants.items():
    print(city_avg[[col]])
    city_avg[col].plot(kind='bar')
    plt.title(f"Average {label} by City")
    plt.xlabel("City")
    plt.ylabel(label)
    plt.show()
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
print(df['AQI_Category'].value_counts())
'''plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='city', y='pm2_5')
plt.title('PM2.5 Distribution Across Cities', fontsize=14, fontweight='bold')
plt.xlabel('City')
plt.ylabel('PM2.5 (µg/m³)')
plt.show()
numerical_cols=['pm10','pm2_5','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide','ozone']
print("Statistics by City")
city_stats=df.groupby('city')[numerical_cols].agg(['mean','median','std']).round(2)
print(city_stats)'''
