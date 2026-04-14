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
df['date']=pd.to_datetime(df['date'])
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
print(df.describe())
plt.figure()
plt.plot(df['date'],df['pm2_5'])
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("PM2.5 Over Time")
plt.xticks(rotation=45)
plt.show()
city_avg = df.groupby('city').mean(numeric_only=True)
print(city_avg[['pm2_5']])
print(city_avg[['pm10']])
print(city_avg[['carbon_monoxide']])
print(city_avg[['nitrogen_dioxide']])
city_avg['pm2_5'].plot(kind='bar')
plt.title("Average PM2.5 by City")
plt.xlabel("City")
plt.ylabel("PM2.5")
plt.show()
city_avg['pm10'].plot(kind='bar')
plt.title("Average PM10 by City")
plt.xlabel("City")
plt.ylabel("PM10")
plt.show()
city_avg['carbon_monoxide'].plot(kind='bar')
plt.title("Average CO by City")
plt.xlabel("City")
plt.ylabel("CO")
plt.show()
city_avg['nitrogen_dioxide'].plot(kind='bar')
plt.title("Average NO2 by City")
plt.xlabel("City")
plt.ylabel("NO2")
plt.show()
'''def AQI_Category(pm25):
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
numerical_cols=['pm10','pm2_5','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide','ozone']
print("Statistics by City")
city_stats=df.groupby('city')[numerical_cols].agg(['mean','median','std']).round(2)
print(city_stats)'''
