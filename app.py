import streamlit as st
from functions import *

# """
# 1. "Train Model" button
# 2. After button clicked, train data
# 3. After training finished, show graph, user can choose want to show train data or test data, also show the evaluation
# 4. Give user to customize the shift period with streamlit slider
# 5. Give feature to user input
# """
st.title("TSLA Close Price Predictor with LSTM and VADER")


col1, col2 = st.columns(2)
sequence = st.select_slider(
    "Set sequence parameter", options=list(range(1, 11)), value=3
)

train_btn = st.button("Train Model", type="primary")


if train_btn:
    with st.spinner("Training data..."):
        y_train, y_test, y_pred_train, y_pred_test, model = train_data(sequence)
    st.session_state.model_data = {
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "model": model,
    }

if "model_data" in st.session_state:
    model_data = st.session_state.model_data
    y_train = model_data["y_train"]
    y_test = model_data["y_test"]
    y_pred_test = model_data["y_pred_test"]
    y_pred_train = model_data["y_pred_train"]
    model = model_data["model"]

    col1, col2 = st.columns(2)
    with col1:
        data_type = st.selectbox(
            "Choose train or test data to show", ["Train", "Test"], 0
        )
    with col2:
        shift = st.select_slider(
            "Choose shift period", options=list(range(-10, 11)), value=0
        )

    if data_type == "Train":
        df = pd.DataFrame(
            data={"Real Data": y_train, "Predict Data": y_pred_train.flatten()}
        )
        df["Predict Data"] = df["Predict Data"].shift(shift)

    elif data_type == "Test":
        df = pd.DataFrame(
            data={"Real Data": y_test, "Predict Data": y_pred_test.flatten()}
        )
        df["Predict Data"] = df["Predict Data"].shift(shift)
    st.line_chart(df, height=500)
    df.dropna(inplace=True)
    mae, mse, rmse, mape, smape, r2 = evaluate(df["Real Data"], df["Predict Data"])
    evaluation_df = pd.DataFrame(
        data=[mae, mse, rmse, mape, smape, r2],
        index=["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "r2"],
        columns=["Score"],
    )
    st.table(evaluation_df)

    st.subheader("LSTM Predictor")
    st.text("Input your VADER score with ascending timeframe")

    input_data = st.data_editor(
        pd.DataFrame(
            columns=["Positive Score", "Negative Score", "Neutral Score"],
            index=list(range(sequence)),
        ),
        column_config={
            "Positive Score": st.column_config.NumberColumn(min_value=0, max_value=1),
            "Negative Score": st.column_config.NumberColumn(min_value=0, max_value=1),
            "Neutral Score": st.column_config.NumberColumn(min_value=0, max_value=1),
        },
    )

    forecast_btn = st.button("Forecast")
    if forecast_btn:
        if input_data.isna().any().any():
            st.error("You must input all fields")
        else:
            dummy_row = pd.DataFrame(input_data.iloc[-1]).T
            print(dummy_row)
            input_data = pd.concat([input_data, dummy_row])
            predict = float(forecast(input_data, sequence, model))
            print(type(predict))

            st.success(f"Predicted Close Price: ${np.round(predict, 2)}")
