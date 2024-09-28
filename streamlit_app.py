import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='Water Level Prediction (LSTM)', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการข้อมูลระดับน้ำและการพยากรณ์ด้วย LSTM")

# ฟังก์ชันสร้างข้อมูลสำหรับ LSTM
def create_dataset(data, look_back=15):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# ฟังก์ชันคำนวณความแม่นยำ
def calculate_accuracy(filled_data, original_data):
    # ต้องใช้เฉพาะตำแหน่งที่มีค่าหายไป
    missing_indexes = original_data[original_data['wl_up'].isna()].index
    actual_values = original_data.loc[missing_indexes, 'wl_up']
    predicted_values = filled_data.loc[missing_indexes, 'wl_up_filled']
    
    # คำนวณ MAE และ RMSE
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# ฟังก์ชันพยากรณ์ค่าระดับน้ำด้วย LSTM
def predict_missing_values(df, model_path, look_back=15):
    # โหลดโมเดล LSTM ที่ฝึกแล้ว
    model = load_model(model_path)
    
    # ฟิต Scaler ด้วยข้อมูลที่มีค่าไม่หายไป
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_not_null = df[df['wl_up'].notnull()]
    scaler.fit(df_not_null[['wl_up']])
    
    # ปรับขนาดข้อมูลทั้งหมดด้วย Scaler ที่ฟิตแล้ว
    df_scaled = df.copy()
    df_scaled['wl_up_scaled'] = scaler.transform(df[['wl_up']])
    
    # เติมค่าหายไปด้วยศูนย์ชั่วคราว เพื่อสร้างลำดับเวลา
    df_scaled['wl_up_filled'] = df_scaled['wl_up_scaled'].fillna(0)
    
    # เตรียมข้อมูลสำหรับ LSTM
    X = []
    for i in range(len(df_scaled) - look_back):
        X.append(df_scaled['wl_up_filled'].iloc[i:i+look_back].values)
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # พยากรณ์เฉพาะตำแหน่งที่ค่าหายไป
    predictions = []
    for i in range(len(X)):
        idx = i + look_back
        if np.isnan(df['wl_up'].iloc[idx]):
            X_input = X[i].reshape(1, look_back, 1)
            y_pred = model.predict(X_input)
            predictions.append(y_pred[0][0])
        else:
            predictions.append(None)
    
    # สร้าง DataFrame สำหรับค่าที่ทำนาย
    df_predictions = df.copy()
    pred_index = df_predictions.iloc[look_back:].index
    df_predictions.loc[pred_index, 'wl_up_predicted'] = predictions
    
    # Inverse scaling สำหรับค่าที่ทำนาย
    df_predictions['wl_up_predicted'] = scaler.inverse_transform(df_predictions[['wl_up_predicted']])
    
    # เติมค่าที่ทำนายกลับเข้าไปในตำแหน่งที่หายไป
    df_predictions['wl_up_filled'] = df_predictions['wl_up']
    df_predictions.loc[df_predictions['wl_up'].isna(), 'wl_up_filled'] = df_predictions.loc[df_predictions['wl_up'].isna(), 'wl_up_predicted']
    
    return df_predictions

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    # โหลดข้อมูลจริง
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # ทำให้ datetime เป็น tz-naive (ไม่มี timezone)
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    
    # ตั้งค่า datetime เป็นดัชนี
    data.set_index('datetime', inplace=True)

    # **เพิ่มการกรองข้อมูลที่ wl_up >= 100**
    data = data[data['wl_up'] >= 100]

    # **เรียงข้อมูลตามวันที่และเวลา**
    data = data.sort_index()

    # **แสดงกราฟของข้อมูลที่กรองและเรียงแล้ว**
    st.subheader("กราฟข้อมูลระดับน้ำที่กรองและเรียงแล้ว")
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['wl_up'], label='Water Level (wl_up)', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Water Level (wl_up)')
    plt.title('Water Level over Time (Filtered and Sorted)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # ตรวจสอบว่ามีค่าที่หายไปหรือไม่
    if data['wl_up'].isnull().any():
        st.write("พบค่าที่หายไปในข้อมูล กำลังทำการเติมค่า...")
        
        # เก็บสำเนาของข้อมูลก่อนเติมค่า
        original_data = data.copy()
        
        # พยากรณ์และเติมค่าที่หายไปด้วยโมเดล LSTM
        filled_data = predict_missing_values(data, "lstm_2024_50epochs.keras")
        
        # คำนวณความแม่นยำ
        calculate_accuracy(filled_data, original_data)
        
        # แสดงกราฟข้อมูลที่ถูกเติม
        plt.figure(figsize=(14, 7))
        
        # ข้อมูลก่อนเติมค่า
        plt.plot(original_data.index, original_data['wl_up'], label='Original Data', color='blue')
        
        # ข้อมูลที่ถูกเติมค่า
        plt.plot(filled_data.index, filled_data['wl_up_filled'], label='Filled Data', color='green')
        
        plt.xlabel('Date')
        plt.ylabel('Water Level (wl_up)')
        plt.title('Water Level over Time with Missing Values Filled')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        
        # แสดงผลลัพธ์การเติมค่าเป็นตาราง
        st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up_filled)')
        st.write(filled_data[['wl_up_filled']])
    else:
        st.write("ไม่พบค่าที่หายไปในข้อมูล ไม่จำเป็นต้องเติมค่า")

