import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Prediksi Harga Laptop",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Aplikasi Prediksi Harga Laptop")
st.write("Masukkan spesifikasi laptop di bawah ini untuk memprediksi harganya.")
st.markdown("---")

# Pakai @st.cache_resource supaya model cuma dilatih sekali saat web dibuka
# Jadi gak lemot setiap kali ganti input
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('data.csv')
        
        cols_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'brand', 'name', 'processor', 
                        'CPU', 'Ram_type', 'ROM', 'ROM_type', 'GPU', 'display_size', 'OS', 'warranty']
        
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing_cols)

        df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').str.replace('TB', '')
        df['Ram'] = df['Ram'].astype(int)

        X = df.drop('price', axis=1)
        y = df['price']
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model
        
    except FileNotFoundError:
        st.error("❌ File 'data.csv' tidak ditemukan di folder ini. Harap masukkan file dataset.")
        return None
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
        return None

model = train_model()

if model:
    st.sidebar.header("⚙️ Masukkan Spesifikasi")

    input_rating = st.sidebar.slider("Spec Rating (Kualitas)", 0, 100, 60, help="Makin tinggi makin bagus. 40-50 (Standard), 60-75 (Menengah), 80+ (Gaming/Pro)")
    input_ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 12, 16, 32, 64], index=1)
    input_width = st.sidebar.number_input("Resolusi Lebar (px)", 800, 4000, 1920)
    input_height = st.sidebar.number_input("Resolusi Tinggi (px)", 600, 3000, 1080)

    if st.button("🔍 Hitung Estimasi Harga"):
        
        data_baru = pd.DataFrame({
            'spec_rating': [input_rating],
            'Ram': [input_ram],
            'resolution_width': [input_width],
            'resolution_height': [input_height]
        })

        prediksi_inr = model.predict(data_baru)[0]

        kurs_idr = 190 
        prediksi_idr = prediksi_inr * kurs_idr

        st.subheader("Hasil Prediksi")

        if prediksi_idr < 0:
            st.warning("⚠️ **Hasil Negatif**")
            st.write(f"Nilai mentah: Rp {prediksi_idr:,.2f}")
        else:
            st.success(f"💰 **Rp {prediksi_idr:,.0f}**")
            
            with st.expander("Lihat Detail Spesifikasi"):
                st.write(f"- **Rating:** {input_rating}")
                st.write(f"- **RAM:** {input_ram} GB")
                st.write(f"- **Resolusi:** {input_width} x {input_height}")

else:
    st.info("Menunggu file 'data.csv'...")