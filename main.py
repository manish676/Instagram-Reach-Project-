import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
import pandas as pd
data = pd.read_csv("Instagram-Reach.csv", encoding = 'latin-1')
print(data.head())

# Let’s analyze the trend of Instagram reach over time using a line chart:
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=data['Date'],
#                          y=data['Instagram reach'],
#                          mode='lines', name='Instagram reach'))
# fig.update_layout(title='Instagram Reach Trend', xaxis_title='Date',
#                   yaxis_title='Instagram Reach')
# fig.show()
#


# Now let’s analyze Instagram reach for each day using a bar chart

#
# fig = go.Figure()
# fig.add_trace(go.Bar(x=data['Date'],
#                      y=data['Instagram reach'],
#                      name='Instagram reach'))
# fig.update_layout(title='Instagram Reach by Day',
#                   xaxis_title='Date',
#                   yaxis_title='Instagram Reach')
# fig.show()


# Now let’s analyze the distribution of Instagram reach using a box plot:

# fig = go.Figure()
# fig.add_trace(go.Box(y=data['Instagram reach'],
#                      name='Instagram reach'))
# fig.update_layout(title='Instagram Reach Box Plot',
#                   yaxis_title='Instagram Reach')
# fig.show()

# Now let’s create a day column and analyze reach based on the days of the week. To create a day column, we can use the dt.day_name() method to extract the day of the week from the Date column:

# import  numpy as np
# import pandas as pd
#
# # Assuming you have already loaded or created your DataFrame 'data'
# # If 'Date' is not in datetime format, convert it
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Check for missing values
# print(data['Date'].isnull().sum())
#
# # Add a new column 'Day' with the day names
# data['Day'] = data['Date'].dt.day_name()
#
# # Display the updated DataFrame
# # print(data.head())


#this is showing this Day  mean   median std


# import numpy as np
# import pandas as pd
#
# # Assuming you have already loaded or created your DataFrame 'data'
# # If 'Date' is not in datetime format, convert it
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Create a new column 'Day' with the day names
# data['Day'] = data['Date'].dt.day_name()
#
# # Group by 'Day' and calculate mean, median, and std for 'Instagram reach'
# day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
#
# # Display the statistics
# print(day_stats)



# Now, let’s create a bar chart to visualize the reach for each day of the week:



# import plotly.graph_objs as go
# import pandas as pd
#
# # Assuming you have loaded or created your DataFrame 'data'
# # ... (Previous code for data processing)
#
# # Convert 'Date' to datetime if not already done
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Create a new column 'Day' with the day names
# data['Day'] = data['Date'].dt.day_name()
#
# # Ensure the 'Day' column is created successfully
# print("Columns in data:", data.columns)
#
# # Group by 'Day' and calculate mean, median, and std for 'Instagram reach'
# try:
#     day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
#     # Display the statistics
#     print(day_stats)
#
#     # Create a bar chart using Plotly
#     fig = go.Figure()
#
#     fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['mean'], name='Mean'))
#     fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['median'], name='Median'))
#     fig.add_trace(go.Bar(x=day_stats['Day'], y=day_stats['std'], name='Standard Deviation'))
#
#     fig.update_layout(title='Instagram Reach by Day of the Week',
#                       xaxis_title='Day',
#                       yaxis_title='Instagram Reach')
#
#     fig.show()
#
# except KeyError as e:
#     print(f"Error: Column '{e.args[0]}' not found. Check the actual column names in your DataFrame.")


#Let’s look at the Trends and Seasonal patterns of Instagram reach:

# from plotly.tools import mpl_to_plotly
# import matplotlib.pyplot as plt
# from statsmodels.tsa.seasonal import seasonal_decompose
#
# data = data[["Date", "Instagram reach"]]
#
# result = seasonal_decompose(data['Instagram reach'],
#                             model='multiplicative',
#                             period=100)
#
# fig = plt.figure()
# fig = result.plot()
#
# fig = mpl_to_plotly(fig)
# fig.show()

#Now here’s how to visualize an autocorrelation plot to find the value of p:

# import pandas as pd
# import matplotlib.pyplot as plt

# Assuming you have loaded or created your DataFrame 'data'
# ... (Previous code for data processing)

# Make sure the 'Date' column is in datetime format
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Plot the autocorrelation
# pd.plotting.autocorrelation_plot(data["Instagram reach"])
# plt.title('Autocorrelation Plot')
# plt.show()


#And now here’s how to visualize a partial autocorrelation plot to find the value of q:

# import pandas as pd
# from statsmodels.graphics.tsaplots import plot_pacf
# import matplotlib.pyplot as plt
#
# # Assuming you have loaded or created your DataFrame 'data'
# # ... (Previous code for data processing)
#
# # Make sure the 'Date' column is in datetime format
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Plot the PACF
# fig, ax = plt.subplots(figsize=(12, 6))
# plot_pacf(data["Instagram reach"], lags=100, ax=ax)
# plt.title('Partial Autocorrelation Function (PACF)')
# plt.show()



#Now here’s how to train a model using SARIMA:
import statsmodels.api as sm

# Assuming you have loaded or created your DataFrame 'data'
# ... (Previous code for data processing)

# Make sure the 'Date' column is in datetime format
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Adjust values for p, d, and q based on your analysis
# p, d, q = 1, 1, 1
#
# # SARIMAX model with seasonality (assuming you have monthly data)
# model = sm.tsa.statespace.SARIMAX(data['Instagram reach'],
#                                   order=(p, d, q),
#                                   seasonal_order=(1, 1, 1, 12))
# model = model.fit()
# print(model.summary())



# import plotly.graph_objs as go
# import pandas as pd
# import statsmodels.api as sm
#
# # Assuming you have loaded or created your DataFrame 'data'
# # ... (Previous code for data processing)
#
# # Assuming you have fitted your SARIMAX model
# p, d, q = 1, 1, 1
# model = sm.tsa.statespace.SARIMAX(data['Instagram reach'],
#                                   order=(p, d, q),
#                                   seasonal_order=(1, 1, 1, 12))
# model = model.fit()
#
# # Assuming you have generated predictions
# predictions = model.predict(start=len(data), end=len(data) + 100)
#
# # Scatter plot for training data
# trace_train = go.Scatter(x=data.index,
#                          y=data["Instagram reach"],
#                          mode="lines",
#                          name="Training Data")
#
# # Scatter plot for predictions
# trace_pred = go.Scatter(x=predictions.index,
#                         y=predictions,
#                         mode="lines",
#                         name="Predictions")
#
# layout = go.Layout(title="Instagram Reach Time Series and Predictions",
#                    xaxis_title="Date",
#                    yaxis_title="Instagram Reach")
#
# fig = go.Figure(data=[trace_train, trace_pred], layout=layout)
# fig.show()


#So this is how we can forecast the reach of an Instagram account using Time Series Forecasting.