import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Saham IHSG", page_icon="📈", layout="wide")

# Judul aplikasi
st.title("📈 Analisis Saham IHSG")
st.markdown("Platform analisis saham Indonesia menggunakan data Yahoo Finance")

# Sidebar untuk input
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Input kode saham
    stock_code = st.text_input(
        "Kode Saham", 
        value="BBCA.JK",
        help="Masukkan kode saham dengan .JK (contoh: BBCA.JK, TLKM.JK, BBRI.JK)"
    )
    
    # Pilihan periode
    period_options = {
        "1 Hari": "1d",
        "1 Pekan": "5d",
        "1 Bulan": "1mo",
        "3 Bulan": "3mo",
        "6 Bulan": "6mo",
        "1 Tahun": "1y",
        "3 Tahun": "3y",
        "5 Tahun": "5y",
        "10 Tahun": "10y"
    }
    
    selected_period = st.selectbox("Periode Waktu", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Tombol analisis
    analyze_button = st.button("🔍 Analisis Saham", type="primary")

# =============================================================================
# SOLUSI: Fungsi Caching untuk mengambil data
# Ini akan menyimpan data selama 10 menit (600 detik) untuk mengurangi panggilan API
# =============================================================================
@st.cache_data(ttl=600)
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        info = stock.info
        
        if df.empty:
            st.error(f"Tidak dapat menemukan data historis untuk {ticker} pada periode {period}. Mungkin kode salah atau delisted.")
            return None, None
        
        # Validasi sederhana untuk info
        if not info or 'regularMarketPrice' not in info or info.get('regularMarketPrice') is None:
           st.warning(f"Tidak dapat mengambil info profil lengkap untuk {ticker}. Info fundamental mungkin tidak lengkap.")
           # Tetap lanjutkan dengan data historis jika ada
        
        return df, info
    except Exception as e:
        st.error(f"Gagal mengambil data dari yfinance: {e}")
        return None, None
# =============================================================================

# Fungsi untuk menghitung indikator teknikal
def calculate_technical_indicators(df):
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Fungsi untuk memberikan insight
def generate_insights(df, info, period_text):
    insights = []
    
    # Analisis trend harga
    current_price = df['Close'].iloc[-1]
    first_price = df['Close'].iloc[0]
    price_change = ((current_price - first_price) / first_price) * 100
    
    if price_change > 0:
        insights.append(f"📈 Harga saham naik {price_change:.2f}% dalam periode {period_text}")
    else:
        insights.append(f"📉 Harga saham turun {abs(price_change):.2f}% dalam periode {period_text}")
    
    # Analisis Moving Average
    if len(df) >= 20 and not pd.isna(df['MA20'].iloc[-1]):
        if current_price > df['MA20'].iloc[-1]:
            insights.append("✅ Harga di atas MA20 - Trend bullish jangka pendek")
        else:
            insights.append("⚠️ Harga di bawah MA20 - Trend bearish jangka pendek")
    
    # Analisis RSI
    if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
        rsi = df['RSI'].iloc[-1]
        if rsi > 70:
            insights.append(f"🔴 RSI {rsi:.2f} - Overbought (kemungkinan koreksi)")
        elif rsi < 30:
            insights.append(f"🟢 RSI {rsi:.2f} - Oversold (kemungkinan rebound)")
        else:
            insights.append(f"🟡 RSI {rsi:.2f} - Kondisi normal")
    
    # Analisis Volume
    avg_volume = df['Volume'].mean()
    recent_volume = df['Volume'].iloc[-5:].mean()
    if recent_volume > avg_volume * 1.5:
        insights.append("📊 Volume trading meningkat signifikan")
    
    # Analisis Fundamental (cek ketersediaan 'info')
    if info:
        if 'marketCap' in info and info['marketCap']:
            try:
                market_cap_billions = info['marketCap'] / 1_000_000_000_000 # Untuk Triliun
                insights.append(f"💰 Market Cap: Rp {market_cap_billions:.2f} Triliun")
            except Exception:
                pass
        
        if 'trailingPE' in info and info['trailingPE']:
            pe_ratio = info['trailingPE']
            if pe_ratio < 15:
                insights.append(f"📊 P/E Ratio {pe_ratio:.2f} - Valuasi menarik")
            elif pe_ratio > 25:
                insights.append(f"📊 P/E Ratio {pe_ratio:.2f} - Valuasi tinggi")
            else:
                insights.append(f"📊 P/E Ratio {pe_ratio:.2f} - Valuasi wajar")
        
        if 'dividendYield' in info and info['dividendYield']:
            div_yield = info['dividendYield'] * 100
            if div_yield > 3:
                insights.append(f"💵 Dividend Yield {div_yield:.2f}% - Dividen menarik")
    
    return insights

# Main aplikasi
if analyze_button:
    st.session_state.last_stock = stock_code
    st.session_state.last_period = period
    st.session_state.last_period_text = selected_period

if 'last_stock' in st.session_state:
    
    # Ambil data dari cache atau panggil API jika cache sudah habis/belum ada
    with st.spinner(f"Mengambil data {st.session_state.last_stock}..."):
        df, info = get_stock_data(st.session_state.last_stock, st.session_state.last_period)

    # Lanjutkan hanya jika data berhasil diambil
    if df is not None and info is not None:
        
        try:
            # Hitung indikator teknikal
            df = calculate_technical_indicators(df)
            
            # Info saham
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Harga Terakhir",
                    f"Rp {df['Close'].iloc[-1]:,.2f}",
                    f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100):.2f}% (1 hari)"
                )
            
            with col2:
                st.metric(
                    "Harga Tertinggi",
                    f"Rp {df['High'].max():,.2f}"
                )
            
            with col3:
                st.metric(
                    "Harga Terendah",
                    f"Rp {df['Low'].min():,.2f}"
                )
            
            with col4:
                st.metric(
                    "Volume Rata-rata",
                    f"{df['Volume'].mean():,.0f}"
                )
            
            # Informasi perusahaan
            st.subheader(f"📊 {info.get('longName', st.session_state.last_stock)}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Sektor:** {info.get('sector', 'N/A')}")
                st.write(f"**Industri:** {info.get('industry', 'N/A')}")
            
            with col2:
                if 'marketCap' in info and info['marketCap']:
                    st.write(f"**Market Cap:** Rp {info['marketCap']/1e12:.2f}T")
                if 'trailingPE' in info and info['trailingPE']:
                    st.write(f"**P/E Ratio:** {info['trailingPE']:.2f}")
            
            with col3:
                if 'dividendYield' in info and info['dividendYield']:
                    st.write(f"**Dividend Yield:** {info['dividendYield']*100:.2f}%")
                if 'beta' in info and info['beta']:
                    st.write(f"**Beta:** {info['beta']:.2f}")
            
            # Grafik harga dengan candlestick
            st.subheader("📈 Grafik Harga Saham")
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('Harga & Moving Averages', 'Volume', 'RSI')
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Harga'
                ),
                row=1, col=1
            )
            
            # Moving Averages
            if 'MA20' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange', width=1)),
                    row=1, col=1
                )
            if 'MA50' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='blue', width=1)),
                    row=1, col=1
                )
            if 'MA200' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(color='red', width=1)),
                    row=1, col=1
                )
            
            # Volume
            colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(
                go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # MACD Chart
            if 'MACD' in df.columns:
                st.subheader("📊 MACD (Moving Average Convergence Divergence)")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red')))
                fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD']-df['Signal'], name='Histogram'))
                fig_macd.update_layout(height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Insights
            st.subheader("💡 Insights & Analisis")
            insights = generate_insights(df, info, st.session_state.last_period_text)
            
            for insight in insights:
                st.write(insight)
            
            # Disclaimer
            st.info("⚠️ **Disclaimer:** Analisis ini hanya untuk tujuan edukasi dan informasi. Bukan merupakan rekomendasi untuk membeli atau menjual saham. Lakukan riset sendiri sebelum berinvestasi.")
            
            # Data tabel
            with st.expander("📋 Lihat Data Historis"):
                st.dataframe(df.tail(50))
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {str(e)}")

    else:
        st.error(f"Gagal memuat data untuk {st.session_state.last_stock}. Pastikan kode saham benar.")

else:
    # Tampilan awal
    st.info("👈 Masukkan kode saham di sidebar dan klik **Analisis Saham** untuk memulai")
    
    st.markdown("""
    ### 📌 Panduan Penggunaan:
    1. Masukkan **kode saham** dengan format `.JK` (contoh: BBCA.JK, TLKM.JK, BBRI.JK)
    2. Pilih **periode waktu** yang ingin dianalisis
    3. Klik tombol **🔍 Analisis Saham**
    
    ### 📊 Fitur Analisis:
    - **Grafik Candlestick** dengan Moving Averages (MA20, MA50, MA200)
    - **Volume Trading** untuk melihat aktivitas perdagangan
    - **RSI (Relative Strength Index)** untuk deteksi overbought/oversold
    - **MACD** untuk analisis momentum
    - **Informasi Fundamental** (P/E Ratio, Market Cap, Dividend Yield)
    - **Insights Otomatis** berdasarkan analisis teknikal dan fundamental
    
    ### 💡 Contoh Kode Saham Populer:
    - **BBCA.JK** - Bank BCA
    - **BBRI.JK** - Bank BRI
    - **TLKM.JK** - Telkom Indonesia
    - **ASII.JK** - Astra International
    - **UNVR.JK** - Unilever Indonesia
    - **BMRI.JK** - Bank Mandiri
    """)