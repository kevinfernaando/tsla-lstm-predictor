import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split


vader_score = pd.read_csv("./data/vader_score.csv", index_col=0)
vader_score.index = pd.to_datetime(vader_score.index).normalize()
vader_score = vader_score.resample("D").median()
vader_score = vader_score.apply(lambda x: x + x.shift()).rename(
    columns={"pos": "Positive Score", "neg": "Negative Score", "neu": "Neutral Score"}
)

tsla = pd.read_csv("./data/tsla.csv", index_col=0)
tsla.index = pd.to_datetime(tsla.index).normalize()

close_scaler = MinMaxScaler()
close_scaler.fit(tsla["Close"].to_frame())

tsla_scaler = MinMaxScaler()
tsla_scaler.fit(tsla)
tsla_scaled = pd.DataFrame(
    tsla_scaler.transform(tsla), index=tsla.index, columns=tsla.columns
)

merged_scaled = (
    pd.concat([tsla_scaled, vader_score], axis=1)
    .interpolate()
    .loc["2023":]
    .sort_index()
)


y_predicts = []
for score in ["Positive Score", "Negative Score", "Neutral Score"]:
    X_train, X_test, y_train, y_test = train_test_split(
        merged_scaled["Close"], merged_scaled[score], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train.values.reshape(-1, 1), y_train)
    equation = f"{model.coef_[0]:.4f} * Close + {model.intercept_:.4f}"
    y_predict = model.predict(merged_scaled["Close"].values.reshape(-1, 1))
    y_predicts.append(y_predict)

predict_data = pd.DataFrame(
    np.array(y_predicts).T,
    columns=[
        "Positive Score Predict",
        "Negative Score Predict",
        "Neutral Score Predict",
    ],
    index=merged_scaled.index,
).merge(merged_scaled["Close"], left_index=True, right_index=True)


def train_data(sequence=10):
    # Prepare input sequences
    def create_sequences(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X.iloc[i : (i + time_steps)].values)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = sequence
    X = predict_data[
        ["Positive Score Predict", "Negative Score Predict", "Neutral Score Predict"]
    ]
    y = predict_data["Close"]

    # Create sequences
    X_seq, y_seq = create_sequences(X, y, time_steps)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )

    # Model architecture
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam")

    # Fit the model
    model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )

    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    return y_train, y_test, y_pred_train, y_pred_test, model


def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    smape = (
        np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))) * 100
    )
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, mape, smape, r2


def forecast(df, sequence, model):
    def create_sequences(X, time_steps):
        Xs = []
        for i in range(len(X) - time_steps):
            Xs.append(X.iloc[i : (i + time_steps)].values)
        return np.array(Xs)

    X_seq = create_sequences(df, sequence).astype(np.float32)
    print(df)
    print(X_seq)
    predict = model.predict(X_seq)
    predict = close_scaler.inverse_transform(predict).flatten()[0]

    return predict
