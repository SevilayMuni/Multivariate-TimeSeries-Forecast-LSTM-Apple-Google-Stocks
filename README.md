
![coverimage](https://www.nyse.com/publicdocs/images/Hero_1150x550_BlueFinChart-02.png)


## 🔮 Multivariate Time Series Forecast via LSTM: Apple and Google Stocks 💰

## A complete example to create robust LSTM models for Apple and Google stock price forecasting.

Every part of the work exemplifies code how to perform following:
    
- Obtain historical data using yfinance module in python.
- Implement feature engineering to create technical indicators.
- Conduct time-series data visualizations to establish trends in stock and relations among variables.
- Generate correlation heatmap for variable selection.
- Prepare datasets for LSTM model; time-window function, train/test split, MinMax Scaling.
- Design LSTM model architecture via Keras Sequential() Class.
- Execute GridSearchCV for Keras model hyperparameter tuning.
- Display best paramters and create several LSTM models.
- Get prediction from model and inverse scale test and predicted data.
- Graph real/predicted stock prices to evaluate model accuracy. 
- Calculate common performance metrics (MAE, MAPE, R2) for forecasting model via Sklearn.
- Compare and contrast different model performances using graphs and metrics tables. 



## Introduction to Model 📚

The stock market has always been a challenging and attractive domain for research with data scientists continuously seeking ways to predict stock prices accurately. With advancements in machine learning and deep learning, new techniques are emerging that show potential for making stock price prediction more reliable. 

Long Short-Term Memory (LSTM) networks, a variant of recurrent neural networks (RNNs), have proven effective for time series forecasting, particularly when dealing with sequential data like stock prices. A time series is simply a sequence of data points indexed in time order.

Nonetheless, univariate time series forecasting, involves using only past stock prices for prediction, can be limiting. Stock price movement is influenced by a variety of factors; thus, multivariate time series forecasting significantly improves model accuracy. By feature engineering based on domain knowledge enables the model to capture the underlying patterns and relationships in the data.

LSTM is designed to overcome the limitations of traditional RNNs by using memory cells and gates (input, forget, and output gates) to control the flow of information. This architecture allows LSTMs to capture long-term dependencies and avoid the vanishing gradient problem, making them more suitable for financial time series, where past data often has long-lasting effects on future prices.
## Dataset and Feature Engineering
Historical dataset acquired from Yahoo Finance has following variables: 

    open, high, low, close, adjusted close
**Feature engineering is the process of selecting and transforming variables that make the prediction model more accurate.**
    
In the project, following technical indicators are utilized:
    
    Garman-Klass Volatility
    Dollar Volume
    On Balance Volume (OBV)
    Moving Average Convergence Divergence (MACD)
    Moving Averages (MAs)

[Read on Technical Indicators](https://www.britannica.com/money/technical-indicator-types)


## Model Building 👩🏻‍💻 
For neural network, dataset prepared as following:
1. Create DF with selected independent variables (based on correlation heatmap) and target variable
2. Split train/test dsets wrt 80% - 20%
3. Scale train and test dsets to avoid prediction errors via MinMaxScaler

```
# n_past = number of step model will look in the past to predict
def createXY(dataset, n_past): 
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

trainX, trainY = createXY(apple_train_scaled, 21)
testX, testY = createXY(apple_test_scaled, 21)
```

Once datasets are prepared, **Keras Sequential** module is used for model architecture.
**GridSearchCV** is utilized for hyperparameter optimization. 

```
# Construct model architecture
def build_2_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(128, return_sequences = True, input_shape = (21, 5)))
    grid_model.add(LSTM(64))
    grid_model.add(Dense(10))
    grid_model.add(Dense(1))
    grid_model.compile(loss = 'mse', optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn = build_2_model, verbose = 1, validation_data = (testX, testY))
```
```
# Define values for hyperparameters
parameters = {'batch_size' : [16, 20, 26], 'epochs' : [10, 15, 20], 'optimizer' : ['adam', 'Adadelta']}
grid_search  = GridSearchCV(estimator = grid_model, param_grid = parameters, cv = 4)
```
```
# Models with different parameter values will be trained and minimize MSE 
grid_search = grid_search.fit(trainX, trainY)
```
```
# Create trained model with best parameter values
apple_model2 = grid_search.best_estimator_.model
```
**The model is ready for test dataset :)** 

## Model Evalution 📈

Step 5: Evaluating Model Performance
After training the model, we can evaluate its performance on the test set using evaluation metrics such as:
Mean Absolute Error (MAE)
Root Mean Absolute Error (MAPE)
R-Squared (R²)

The predicted stock prices can be compared to actual prices to see how well the model performs. Additionally, visualizing the results using line plots helps in understanding the accuracy of the predictions over time.


IMAGES

| Model | MAE | MAPE | R2 |
| --- | --- | --- | --- |
| `Apple: 1st Model` | 3.358 | 0.021 | 0.974 |
| `Apple: 2nd Model` | 2.777 | 0.018 | 0.983 |
| `Google: 1st Model` | 5.110 | 0.037 | 0.924 |
| `Google: 2nd Model` | 2.625| 0.021| 0.980 |
| `Google: 3rd Model` | 4.136| 0.031 | 0.955 |

🎉 **For both Apple and Google stock price prediction, respective 2nd models yielded best result graph and evaluation metrics.** 🎉

IMAGES
## Model Usage 🤖
Once the model performs accurately, we can predict future stock prices based on the latest data. The process involves feeding the model new input sequences like the most recent 31 days and obtaining a forecasted stock price for the next day or future periods.
## Challenges and Limitations
While my LSTM models has robust performance in stock price prediction, there are challenges: champion models for both Apple and Google stock price forecasting challenged in predicting near past (see model performance graphs)

    How to overcome the challenge?
    a. Increase the data size (however, not applicable to Google stock)
    b. Revise feature engineering; add other technical indicators 
    c. Change set of independent variables and n_past
    d. Improve LSTM model design and GridSearchCV (e.g search for more parameters or values)

Nevertheless, stock price forecast by the LSTM model has its intrinsic limitations:

**Volatility and Noise:** Stock prices are highly volatile and influenced by numerous factors such as news, earnings reports, or macroeconomic events. This noise can make predictions less reliable.

**Overfitting:** LSTM models can overfit training data if not appropriately regularized, especially when dealing with a small dataset or noisy data.

## Conclusion
- Multivariate time series forecasting using LSTM is a powerful method for stock price prediction. It offers the ability to capture complex, long-term dependencies in stock market data. 

- However, it’s essential to account for the challenges of working with financial data and ensure robust model validation to avoid overfitting.
## Warning

This work is not investment advice. It is merely data science project.
## License 🔐
The project is licensed under the MIT License.
## Contact 📩
For any questions or inquiries, feel free to reach out:
- **Email:** sevilaymunire68@gmail.com
- **LinkedIn:** [Sevilay Munire Girgin](www.linkedin.com/in/sevilay-munire-girgin-8902a7159)
Thank you for visiting my project repository. Happy and accurate predicting! 💕
## References 🗂

- Andrés, D., & Andrés, D. (2023, June 24). Error Metrics for Time Series Forecasting - ML pills. ML Pills - Machine Learning Pills. https://mlpills.dev/time-series/error-metrics-for-time-series-forecasting/

- Artificial Intelligence Training: Unlocking the Power of AI with Expert Guided Programs. https://mmcalumni.ca/blog/advanced-techniques-and-strategies-for-effective-artificial-intelligence-training-and-development

- Faressayah. (2023, January 31). 📊Stock Market Analysis 📈 + Prediction using LSTM. Kaggle. https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm

- Introduction to Deep Learning with TensorFlow | Towards AI. https://towardsai.net/p/deep-learning/introduction-to-deep-learning-with-tensorflow

- Jiang, L., Huang, Q., & He, G. (2024). Predicting the Remaining Useful Life of Lithium-Ion Batteries Using 10 Random Data Points and a Flexible Parallel Neural Network. Energies, 17(7), 1695.

- Sasakitetsuya. (2022, April 23). Multivariate Time Series Forecasting with LSTMs. Kaggle. https://www.kaggle.com/code/sasakitetsuya/multivariate-time-series-forecasting-with-lstms

- Sksujanislam. (2023, October 13). MULTIVARIATE TIME SERIES FORECASTING USING LSTM - Sksujanislam - Medium. Medium. https://medium.com/@786sksujanislam786/multivariate-time-series-forecasting-using-lstm-4f8a9d32a509

- Wadaskar, G., Bopanwar, V., Urade, P., Upganlawar, S., & Shende, P. R. (2023). Handwritten Character Recognition. International Journal for Research in Applied Science and Engineering Technology. https://doi.org/10.22214/ijraset.2023.57366



# Multivariate-Time-Series-Forecast-LSTM-Apple-Google-Stocks
# Multivariate-Time-Series-Forecast-LSTM-Apple-Google-Stocks
# Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks
# Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks
# Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks
# Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks
# Multivariate-TimeSeries-Forecast-LSTM-Apple-Google-Stocks
