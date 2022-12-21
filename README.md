# Multivariate Time Series Electricity Consumption Forecasting using ARIMAX, SARIMAX, and RNN-based Deep Learning Models

- Promoted energy conservation by developing time-series forecasting models to forecast daily-average electricity consumption patterns of a residential household.
- Incorporated weather variables and day category (weekday, weekend, vacation day, and COVID lockdown) in the prediction process.
- The optimal model is a RNN-based model (GRU) with **MSE ~ 0.44**.

## Methodology
![methodology](/images/methodology.png)

### Dataset
- The dataset is from [Kaggle](https://www.kaggle.com/datasets/srinuti/residential-power-usage-3years-data-timeseries) and contain hourly electric energy consumption (kWh) from January 2016 to August 2020 of a two-storied house located in Houston, Texas.
- The dataset also includes historical **weather reports** and **day category** (weekday, weekend, vacation day, and COVID lockdown).
![dataset](/images/dataset.png)

### Exploratory Data Analysis
- The dataset’s frequency was converted into the daily frequency to simplify the forecasting task. This led to the downsampling of the dataset from hourly electricity consumption to daily-average electricity consumption
- As seen seasonal decomposition graph, **seasonality** is present in the  time series, that is, every year, **highest consumption can be observed around February to November**, which also happens to be the summer season of Houston, Texas.
![dataset](/images/eda1.png)

