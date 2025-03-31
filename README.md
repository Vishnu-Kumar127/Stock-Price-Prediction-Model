# Stock Price Prediction Model

This repository contains a stock price prediction model using various machine learning and deep learning techniques. The project includes data preprocessing, sentiment analysis, and multiple models such as LSTM, GRU, and LightGBM to predict stock prices.

## Project Structure

### Directories:
- **data/**: Contains datasets used for training and testing the models.
- **models/**: Stores trained model files for predictions.

### Jupyter Notebooks:
1. **2.sentiment_analysis.ipynb** - Performs sentiment analysis on news data to derive sentiment scores that influence stock prices.
2. **3.data_prep.ipynb** - Prepares and preprocesses data, including cleaning, normalization, and feature engineering.
3. **4.LSTM.ipynb** - Implements a Long Short-Term Memory (LSTM) model for time series forecasting of stock prices.
4. **5.GRU.ipynb** - Implements a Gated Recurrent Unit (GRU) model as an alternative to LSTM for stock prediction.
5. **6_LIGHTGBM.ipynb** - Uses LightGBM, a gradient boosting algorithm, for stock price forecasting.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Vishnu-Kumar127/Stock-Price-Prediction-Model.git
   ```
2. Install dependencies (if required):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks in sequence to perform sentiment analysis, data preprocessing, and train the models.

## Dependencies
- Python
- Pandas
- NumPy
- TensorFlow/Keras
- Scikit-learn
- LightGBM
- Matplotlib
- NLTK (for sentiment analysis)

## License
This project is licensed under the MIT License.

## Author
[Vishnu Kumar](https://github.com/Vishnu-Kumar127)

