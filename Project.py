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
corr=df.corr(numeric_only=True)
print(corr)
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)),corr.columns,rotation=45)
plt.yticks(range(len(corr.columns)),corr.columns)
plt.title("Correlation Matrix")
plt.show()
for col,label in pollutants.items():
    plt.figure()
    sns.boxplot(x='city',y=col,data=df)
    plt.title(f"{label} Boxplot")
    plt.show() 
df[['pm10','pm2_5','carbon_monoxide','nitrogen_dioxide']].hist(figsize=(10,8))
plt.show()
def get_season(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Summer'
    elif month in [6,7,8]:
        return 'Monsoon'
    else:
        return 'Post-monsoon'
df['season']=df['month'].apply(get_season)
season_avg=df.groupby('season').mean(numeric_only=True)
print(season_avg['pm2_5'])
top_pollution=df.sort_values(by='pm2_5',ascending=False).head(10)
print(top_pollution[['date','city','pm2_5']]) 
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
df['AQI_Category'].value_counts().plot(kind='bar')
plt.title("AQI Category Distribution")
plt.show()
