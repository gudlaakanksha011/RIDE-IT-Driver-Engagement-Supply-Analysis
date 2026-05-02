# -*- coding: utf-8 -*-
"""Rideit_driver_engagement_Analysis.ipynb

Original file is located at
    https://colab.research.google.com/drive/1ZrtAFmHDYmIF-IZQjrnP20bFpYPlcQtD

# 1. Data Merging and Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
drivers = pd.read_csv('/content/rideit_drivers.csv')
drivers

activity = pd.read_csv('/content/rideit_drivers_activity.csv')
activity

# Merge on Id_driver
df = pd.merge(activity, drivers, on='id_driver', how='inner')

# Convert dates to datetime objects for time-based trends
df['active_date'] = pd.to_datetime(df['active_date'])
df['date_registration'] = pd.to_datetime(df['date_registration'])

"""# 2. Key Insights & Metrics Calculation

A. Conversion Rates (Rides to Offers Ratio)
"""

total_offers = df['offers'].sum()
total_rides = df['rides'].sum()
conversion_rate = (total_rides / total_offers) * 100
print(f"Offer to Ride Conversion: {conversion_rate:.2f}%")

# Booking to Ride Success - Goal: ~82%
booking_success = (df['rides'].sum() / df['bookings'].sum()) * 100
print(f"Global Booking Success Rate: {booking_success:.2f}%")

"""B. Day-Wise Trends (Heatmap Data)"""

# Extract day of week
df['day_of_week'] = df['active_date'].dt.day_name()
day_stats = df.groupby('day_of_week')[['offers', 'bookings', 'rides']].sum()
# Sort by calendar order
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_stats = day_stats.reindex(days)

day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
heatmap_data = df.groupby('day_of_week')[['offers', 'bookings', 'rides']].sum().reindex(day_order).T
plt.figure(figsize=(10, 4))
sns.heatmap(heatmap_data, annot=True, fmt=",d", cmap="YlGnBu")
plt.title('Day Wise Measures: Activity Heatmap')
plt.show()

"""C. Segmentation by Country and Service"""

# Driver Count by Country and Service Type
# This replicates the 'Country Code / Service Type' bar chart in the PPT
driver_segments = drivers.groupby(['country_code', 'service_type'])['id_driver'].nunique().reset_index()
driver_segments.columns = ['country', 'service', 'unique_Drivers']

print("--- Driver Distribution ---")
print(driver_segments)

# 3. Performance Metrics by Segment
# Calculating total rides and average rating for each group
segment_performance = df.groupby(['country_code', 'service_type']).agg({
    'rides': 'sum',
    'driver_rating': 'mean',
    'id_driver': 'nunique'
}).reset_index()

# Calculate Rides Per Driver to identify efficiency[cite: 1]
segment_performance['rides_Per_Driver'] = segment_performance['rides'] / segment_performance['id_driver']

print("\n--- Segment Performance ---")
print(segment_performance)

# 4: Country Code / Service Type[cite: 2]
plt.figure(figsize=(8, 5))
segment_counts = drivers.groupby(['country_code', 'service_type']).size().reset_index(name='Count')
sns.barplot(data=segment_counts, x='country_code', y='Count', hue='service_type')
plt.title('Driver Distribution by Country and Service Type')
plt.show()

# D. Rides to Offers Ratio (Efficiency Line Chart)
daily_stats = df.groupby('day_of_week').agg({'rides':'sum', 'offers':'sum'}).reindex(day_order)
daily_stats['ratio'] = (daily_stats['rides'] / daily_stats['offers']) * 100
plt.figure(figsize=(8, 4))
plt.plot(daily_stats.index, daily_stats['ratio'], marker='o', color='tab:blue')
plt.title('Rides to Offers Ratio (%) by Day')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""# 3. Visualizing Trends and Patterns

Yearly Drop Analysis (COVID-19 Impact)
"""

df['year'] = df['date_registration'].dt.year
yearly_rides = df.groupby('year')['rides'].sum()
ride_percentage = (yearly_rides / yearly_rides.sum()) * 100

ride_percentage.plot(kind='line', marker='o', title='Year wise ride %')
plt.ylabel('% of Total Rides')
plt.show()

"""# 4.Rating Wise Employee Count

"""

def categorize_rating(r):
    if r >= 4.5: return 'Top'
    elif r >= 3.0: return 'Above Average'
    elif r >= 2.0: return 'Average'
    else: return 'Below Average'

drivers['rating_cat'] = drivers['driver_rating'].apply(categorize_rating)
rating_counts = drivers['rating_cat'].value_counts().reindex(['Top', 'Above Average', 'Average', 'Below Average'])
plt.figure(figsize=(7, 5))
rating_counts.plot(kind='bar', color='steelblue')
plt.title('Rating Wise Employee Count')
plt.show()

"""# 5.Booking and Cancellation Trends

# Weekly Passenger vs Driver Cancellations

"""

cancel_stats = df.groupby('day_of_week').agg({
    'bookings_cancelled_by_passenger': 'sum',
    'bookings_cancelled_by_driver': 'sum'
}).reindex(day_order)
plt.figure(figsize=(8, 4))
plt.plot(cancel_stats.index, cancel_stats['bookings_cancelled_by_passenger'], label='Passenger Cancel', marker='s')
plt.plot(cancel_stats.index, cancel_stats['bookings_cancelled_by_driver'], label='Driver Cancel', marker='x')
plt.title('Weekly Cancellation Trends')
plt.legend()
plt.show()

"""# 6. Top 10 Performers (Gold Level Count)[cite: 2]

"""

top_10 = drivers.nlargest(10, 'gold_level_count')
plt.figure(figsize=(10, 5))
sns.barplot(data=top_10, x='id_driver', y='gold_level_count', hue='country_code', dodge=False)
plt.title('Top 10 Performers by Gold Level Count (All Germany)')
plt.show()

"""# 7.FINAL INSIGHTS FOR RECOMMENDATIONS


Find the 10 drivers with low ratings to address challenges?
"""

low_rated_drivers = drivers[drivers['rating_cat'] == 'Below Average']['id_driver'].tolist()
print(f"Drivers to contact regarding challenges: {low_rated_drivers}")

"""# ADVANCED ANALYSIS & STRATEGIC INSIGHTS

# A. Marketing Effectiveness (Statistical Proof)

We compare the performance of drivers who opted into marketing vs those who didn't
"""

marketing_impact = df.groupby('receive_marketing').agg({
    'rides': 'mean',
    'gold_level_count': 'mean',
    'driver_rating': 'mean'
}).reset_index()

print("--- Marketing Impact Analysis ---")
print(marketing_impact)

"""# B. Driver Churn Analysis (Retention)

We identify drivers who haven't completed a ride in the most recent 30 days of the data.
"""

max_date = df['active_date'].max()
churn_cutoff = max_date - pd.Timedelta(days=30)

driver_last_active = df.groupby('id_driver')['active_date'].max().reset_index()
driver_last_active['is_churned'] = driver_last_active['active_date'] < churn_cutoff

churn_rate = (driver_last_active['is_churned'].sum() / len(driver_last_active)) * 100
print(f"\nOverall Driver Churn Rate (Last 30 Days): {churn_rate:.2f}%")

"""# C. Supply-Demand Gap (Burnout Risk)


Analyzing the number of Offers per unique active driver per day.
"""

daily_active_drivers = df.groupby('day_of_week')['id_driver'].nunique()
daily_offers = df.groupby('day_of_week')['offers'].sum()
offers_per_driver = (daily_offers / daily_active_drivers).reindex(day_order)

print("\n--- Offers Per Active Driver (Demand Pressure) ---")
print(offers_per_driver)

"""# D. Correlation Matrix (Identifying Key Drivers of Success)


Checking how ratings, gold status, and activity interact.

"""

corr_matrix = df[['rides', 'offers', 'bookings', 'driver_rating', 'gold_level_count']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0)
plt.title('Correlation Analysis: What Drives Rides?')
plt.show()