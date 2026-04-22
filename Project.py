import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
df=pd.read_csv("Indian Air Quality Trends - 2023-2024.csv")
print("First 5 Entries: \n",df.head())
print("Rows and Columns: ",df.shape)
print("\nDataset Info:")
print(df.info())
print("Missing Values:\n",df.isnull().sum())
print("Duplicate Values:",df.duplicated().sum())
cities=df['city'].unique()
print(cities)
df['date']=pd.to_datetime(df['date'])
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
numerical_cols = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 
                  'sulphur_dioxide', 'ozone']
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
print(df.describe())
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)
plt.figure(figsize=(16, 8))
for i, city in enumerate(cities, 1):
    plt.subplot(2, 3, i)  
    city_data = df[df['city'] == city]
    plt.plot(city_data['date'], city_data['pm2_5'])
    plt.title(f"{city} - PM2.5 Trend")
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.axhline(y=60,color='red',label='Safe Limit')
    plt.legend()
plt.tight_layout()
plt.show()
city_avg = df.groupby('city').mean(numeric_only=True)
plt.figure(figsize=(16,8))
for i,col in enumerate(numerical_cols,1):
    print(city_avg[[col]])
    plt.subplot(2,3,i)
    city_avg[col].plot(kind='bar')
    plt.title(f"Average {col} by City")
    plt.xlabel("City")
    plt.ylabel(col)
plt.tight_layout()
plt.show()
monthly_avg = df.groupby('month').mean(numeric_only=True)
plt.figure(figsize=(16,8))
for i,col in enumerate(numerical_cols,1):
    print(monthly_avg[[col]])
    plt.subplot(2,3,i)
    plt.plot(monthly_avg.index,monthly_avg[col])
    plt.title(f"Average {col} by Month")
    plt.xlabel("Month")
    plt.ylabel(col)
plt.tight_layout()
plt.show()
plt.figure(figsize=(16,8))
for i,col in enumerate(numerical_cols,1):
    plt.subplot(2,3,i)
    sns.boxplot(x='city',y=col,data=df)
    plt.title(f"{col} Boxplot")
plt.tight_layout()
plt.show()
plt.figure(figsize=(16,8))
for i,col in enumerate(numerical_cols,1):
    plt.subplot(2,3,i)
    sns.histplot(df[col], bins=30,kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Between Air Pollutants', fontsize=14, fontweight='bold')
plt.show()
X=df[['pm10']]
Y=df['pm2_5']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("MAE: ",mae)
print("R2 Score: ",r2)
plt.figure()
plt.scatter(X_test, Y_test, label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted")
plt.legend()
plt.title("Linear Regression Trend Prediction")
plt.xlabel("PM10")
plt.ylabel("PM2.5")
plt.title('Simple Linear Regression: PM2.5 vs PM10')
plt.show()
top_pollution=df.sort_values(by='pm2_5',ascending=False).head(5)
print(top_pollution[['date','city','pm2_5']]) 
print(df['AQI_Category'].value_counts())
worst_months = df.groupby(['city', 'month'])['pm2_5'].mean().groupby('city').idxmax()
print("\nWorst month by city:")
for city, month in worst_months.items():
    print(f"  {city}: Month {month[1]}")
print(f"\nOverall average PM2.5 across all cities: {df['pm2_5'].mean():.1f}")
print(f"Percentage of days with PM2.5 > 60 (safe limit): {(df['pm2_5'] > 60).mean()*100:.1f}%")
