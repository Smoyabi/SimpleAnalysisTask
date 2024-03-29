import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#loading csv files into a dataframe
df1=pd.read_csv('cleaned_angers.csv')
df2=pd.read_csv('mumbai_cleaned.csv')
df3=pd.read_csv('Nairobi-Last.csv')
df4=pd.read_csv('nakuruanalysis.csv')
df5=pd.read_csv('Joburg.csv')
df6=pd.read_csv('cleaned_texas.csv')
df7=pd.read_csv('modified_file.csv')

print("DataFrame 1:")
print(df1.head())

print("DataFrame 2:")
print(df2.head())

print("DataFrame 3:")
print(df3.head())

print("DataFrame 4:")
print(df4.head())

print("DataFrame 5:")
print(df5.head())

print("DataFrame 6:")
print(df6.head())

print("DataFrame 7:")
print(df7.head())

# Display information about each DataFrame
print("DataFrame 1 Info:")
print(df1.info())

print("\nDataFrame 2 Info:")
print(df2.info())

print("\nDataFrame 3 Info:")
print(df3.info())

print("\nDataFrame 4 Info:")
print(df4.info())

print("\nDataFrame 5 Info:")
print(df5.info())

print("\nDataFrame 6 Info:")
print(df6.info())

print("\nDataFrame 7 Info:")
print(df7.info())

# Visualize data against time zones for DataFrame 1
plt.figure(figsize=(10, 6))
sns.lineplot(data=df1, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df1['datetimeLocal'], df1['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()

# Visualize data against time zones for DataFrame 2
plt.figure(figsize=(10, 6))
sns.lineplot(data=df2, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df2['datetimeLocal'], df2['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()

# Visualize data against time zones for DataFrame 3
plt.figure(figsize=(10, 6))
sns.lineplot(data=df3, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df3['datetimeLocal'], df3['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()

# Visualize data against time zones for DataFrame 4
plt.figure(figsize=(10, 6))
sns.lineplot(data=df4, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df4['datetimeLocal'], df4['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()


# Visualize data against time zones for DataFrame 5
plt.figure(figsize=(10, 6))
sns.lineplot(data=df5, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df5['datetimeLocal'], df5['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()

# Visualize data against time zones for DataFrame 6
plt.figure(figsize=(10, 6))
sns.lineplot(data=df6, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df6['datetimeLocal'], df6['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()

# Visualize data against time zones for DataFrame 7
plt.figure(figsize=(10, 6))
sns.lineplot(data=df7, x='datetimeLocal', y='value', hue='location_id')
plt.plot(df7['datetimeLocal'], df7['value'], label='Value')
plt.xlabel('Datetime')
plt.ylabel('Value')
plt.title('Air Quality Trends Across Different Regions - DataFrame 1')
plt.legend()
plt.show()

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming 'value' is the target variable for DataFrame 1 
X = df1[['value']]
y = df1['value']

# Split data into training and testing sets for DataFrame 1
X_train_df1, X_test_df1, y_train_df1, y_test_df1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model for DataFrame 1
model_df1 = LinearRegression()
model_df1.fit(X_train_df1, y_train_df1)

# Make predictions for DataFrame 1
predictions_df1 = model_df1.predict(X_test_df1)

# Evaluate model performance for DataFrame 1
mse_df1 = mean_squared_error(y_test_df1, predictions_df1)
print('Mean Squared Error (DataFrame 1):', mse_df1)

# Assuming 'value' is the target variable for DataFrame 2 (df2)
X = df2[['value']]
y = df2['value']

# Split data into training and testing sets for DataFrame 2
X_train_df2, X_test_df2, y_train_df2, y_test_df2 = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model for DataFrame 2
model_df2 = LinearRegression()
model_df2.fit(X_train_df2, y_train_df2)

# Make predictions for DataFrame 2
predictions_df2 = model_df2.predict(X_test_df2)

# Evaluate model performance for DataFrame 2
mse_df2 = mean_squared_error(y_test_df2, predictions_df2)
print('Mean Squared Error (DataFrame 2):', mse_df2)

X = df3[['value']] 
y = df3['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse_df3 = mean_squared_error(y_test, predictions)
print('Mean Squared Error (DataFrame 3):', mse_df3)

X = df4[['value']] 
y = df4['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse_df4 = mean_squared_error(y_test, predictions)
print('Mean Squared Error(DataFrame 4):', mse_df4)

X = df5[['value']] 
y = df5['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse_df5 = mean_squared_error(y_test, predictions)
print('Mean Squared Error(DataFrame 5):', mse_df5)

X = df6[['value']]  
y = df6['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse_df6 = mean_squared_error(y_test, predictions)
print('Mean Squared Error(DataFrame 6):', mse_df6)

X = df7[['value']]  # Select relevant features
y = df7['value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model performance
mse_df7= mean_squared_error(y_test, predictions)
print('Mean Squared Error(DataFrame 7):', mse_df7)


# Create a DataFrame to store MSE values and location IDs
mse_data = {
    'Location ID': [df1['location_id'].iloc[0], df2['location_id'].iloc[0], df3['location_id'].iloc[0], df4['location_id'].iloc[0], df5['location_id'].iloc[0], df6['location_id'].iloc[0], df7['location_id'].iloc[0]],
    'MSE': [mse_df1, mse_df2, mse_df3, mse_df4, mse_df5, mse_df6, mse_df7]
}
mse_df = pd.DataFrame(mse_data)

# Export the DataFrame to a CSV file
mse_df.to_csv('mse_comparison.csv', index=False)
