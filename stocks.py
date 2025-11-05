import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ... (setelah semua import)

# --- FUNGSI CACHE UNTUK YFINANCE ---
# Kita cache data untuk 10 menit (600 detik) atau 1 jam (3600 detik)
# Ini akan mengurangi panggilan API secara drastis

# --- FUNGSI HELPER BARU UNTUK FORMAT ANGKA ---
def format_value(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        if x == 0:
            return "0"
        if abs(x) >= 1_000_000_000_000:
            return f'{x/1_000_000_000_000:.2f}T'
        if abs(x) >= 1_000_000_000:
            return f'{x/1_000_000_000:.2f}B'
        if abs(x) >= 1_000_000:
            return f'{x/1_000_000:.2f}M'
        if abs(x) >= 1_000:
            return f'{x/1_000:.2f}K'
        return f'{x:,.0f}'
    return x

@st.cache_data(ttl=600) # Cache data histori selama 10 menit
def get_stock_history(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        return None
    return df

@st.cache_data(ttl=3600) # Cache info perusahaan selama 1 jam
def get_stock_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        return None # Kembalikan None jika info gagal diambil

@st.cache_data(ttl=3600) # Cache data keuangan selama 1 jam
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        financials = {
            'annual': {
                'income': stock.financials,
                'balance': stock.balance_sheet,
                'cashflow': stock.cashflow
            },
            'quarterly': {
                'income': stock.quarterly_financials,
                'balance': stock.quarterly_balance_sheet,
                'cashflow': stock.quarterly_cashflow
            }
        }
        return financials
    except Exception as e:
        return None

@st.cache_data(ttl=3600) # Cache data holders selama 1 jam
def get_holders_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        holders = {
            'major': stock.major_holders,
            'institutional': stock.institutional_holders
        }
        return holders
    except Exception as e:
        return None

@st.cache_data(ttl=3600) # Cache data rekomendasi selama 1 jam
def get_recommendations_data(ticker):
    try:
        return yf.Ticker(ticker).recommendations
    except Exception as e:
        return None

# --- AKHIR FUNGSI CACHE ---

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Saham IHSG Pro", page_icon="📈", layout="wide")

# CSS Custom
st.markdown("""
<style>
    .big-metric {font-size: 24px; font-weight: bold;}
    .insight-box {padding: 15px; border-radius: 5px; margin: 10px 0;}
    .bullish {background-color: #d4edda; border-left: 5px solid #28a745; color: #000000;} /* Menambahkan teks hitam */
    .bearish {background-color: #f8d7da; border-left: 5px solid #dc3545; color: #000000;} /* Menambahkan teks hitam */
    .neutral {background-color: #fff3cd; border-left: 5px solid #ffc107; color: #000000;} /* Menambahkan teks hitam */
</style>
""", unsafe_allow_html=True)

# Judul aplikasi
st.title("📈 Analisis Saham IHSG - Professional Edition")
st.markdown("Platform analisis saham Indonesia lengkap dengan prediksi pergerakan")

# Sidebar untuk input
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    stock_code = st.text_input(
        "Kode Saham", 
        value="BBCA.JK",
        help="Masukkan kode saham dengan .JK (contoh: BBCA.JK, TLKM.JK)"
    )
    
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
    
    selected_period = st.selectbox("Periode Waktu", list(period_options.keys()), index=5)
    period = period_options[selected_period]
    
    analyze_button = st.button("🔍 Analisis Saham", type="primary")
    
    st.markdown("---")
    st.markdown("### 💡 Contoh Saham")
    st.markdown("""
    - **BBCA.JK** - Bank BCA
    - **BBRI.JK** - Bank BRI
    - **TLKM.JK** - Telkom
    - **ASII.JK** - Astra Int'l
    - **UNVR.JK** - Unilever
    """)

# Fungsi untuk menghitung indikator teknikal lengkap
def calculate_all_indicators(df):
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # EMA
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr)
    minus_di = abs(100 * (minus_dm.rolling(14).sum() / tr))
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df['ADX'] = dx.rolling(14).mean()
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Williams %R
    df['Williams_%R'] = ((high_14 - df['Close']) / (high_14 - low_14)) * -100
    
    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    
    # MFI (Money Flow Index)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    df['MFI'] = mfi
    
    return df

# Fungsi prediksi sederhana
def simple_prediction(df, info):
    signals = []
    scores = 0
    max_score = 0
    
    current_price = df['Close'].iloc[-1]
    
    # 1. Analisis Trend (MA)
    max_score += 3
    if len(df) >= 50:
        if current_price > df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
            signals.append(("✅ Strong Uptrend", "MA menunjukkan trend naik kuat", 2))
            scores += 2
        elif current_price > df['MA20'].iloc[-1]:
            signals.append(("🟢 Uptrend", "Harga di atas MA20", 1))
            scores += 1
        elif current_price < df['MA50'].iloc[-1]:
            signals.append(("🔴 Downtrend", "Harga di bawah MA50", -2))
            scores -= 2
        else:
            signals.append(("🟡 Sideways", "Trend mendatar", 0))
    
    # 2. RSI Analysis
    max_score += 2
    if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signals.append(("🟢 RSI Oversold", f"RSI {rsi:.1f} - Potensi rebound", 2))
            scores += 2
        elif rsi > 70:
            signals.append(("🔴 RSI Overbought", f"RSI {rsi:.1f} - Potensi koreksi", -2))
            scores -= 2
        elif 40 < rsi < 60:
            signals.append(("🟡 RSI Netral", f"RSI {rsi:.1f} - Kondisi normal", 0))
        else:
            signals.append(("⚪ RSI", f"RSI {rsi:.1f}", 0))
    
    # 3. MACD Signal
    max_score += 2
    if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_Signal'].iloc[-1]
        if macd > signal and macd > 0:
            signals.append(("✅ MACD Bullish", "MACD di atas signal line", 2))
            scores += 2
        elif macd < signal and macd < 0:
            signals.append(("🔴 MACD Bearish", "MACD di bawah signal line", -2))
            scores -= 2
        else:
            signals.append(("🟡 MACD Mixed", "Sinyal campuran", 0))
    
    # 4. Bollinger Bands
    max_score += 2
    if 'BB_Lower' in df.columns:
        if current_price < df['BB_Lower'].iloc[-1]:
            signals.append(("🟢 BB Oversold", "Harga di bawah BB Lower", 2))
            scores += 2
        elif current_price > df['BB_Upper'].iloc[-1]:
            signals.append(("🔴 BB Overbought", "Harga di atas BB Upper", -2))
            scores -= 2
    
    # 5. Volume Analysis
    max_score += 1
    avg_volume = df['Volume'].mean()
    recent_volume = df['Volume'].iloc[-5:].mean()
    if recent_volume > avg_volume * 1.5:
        signals.append(("📊 Volume Tinggi", "Aktivitas trading meningkat", 1))
        scores += 1
    elif recent_volume < avg_volume * 0.5:
        signals.append(("📉 Volume Rendah", "Aktivitas trading menurun", 0))
    
    # 6. Stochastic
    max_score += 2
    if '%K' in df.columns:
        k = df['%K'].iloc[-1]
        if k < 20:
            signals.append(("🟢 Stochastic Oversold", f"%K {k:.1f}", 1))
            scores += 1
        elif k > 80:
            signals.append(("🔴 Stochastic Overbought", f"%K {k:.1f}", -1))
            scores -= 1
    
    # 7. ADX Trend Strength
    max_score += 1
    if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]):
        adx = df['ADX'].iloc[-1]
        if adx > 25:
            signals.append(("💪 Trend Kuat", f"ADX {adx:.1f}", 1))
            scores += 1
        else:
            signals.append(("😐 Trend Lemah", f"ADX {adx:.1f}", 0))
    
    # 8. Price Momentum
    max_score += 2
    price_change_5d = ((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100) if len(df) >= 5 else 0
    if price_change_5d > 5:
        signals.append(("🚀 Momentum Kuat", f"Naik {price_change_5d:.1f}% (5 hari)", 2))
        scores += 2
    elif price_change_5d < -5:
        signals.append(("⬇️ Momentum Turun", f"Turun {abs(price_change_5d):.1f}% (5 hari)", -2))
        scores -= 2
    
    # Hitung skor persentase
    score_percentage = ((scores + abs(min(0, scores))) / (max_score * 2)) * 100
    
    # Prediksi
    if scores >= 8:
        prediction = "STRONG BUY 🚀"
        color = "bullish"
        recommendation = "Sinyal beli sangat kuat. Momentum bullish terlihat jelas."
    elif scores >= 4:
        prediction = "BUY 📈"
        color = "bullish"
        recommendation = "Sinyal beli. Trend menunjukkan potensi kenaikan."
    elif scores >= 0:
        prediction = "HOLD ⏸️"
        color = "neutral"
        recommendation = "Tahan posisi. Tunggu konfirmasi sinyal lebih lanjut."
    elif scores >= -4:
        prediction = "SELL 📉"
        color = "bearish"
        recommendation = "Sinyal jual. Pertimbangkan untuk keluar atau cut loss."
    else:
        prediction = "STRONG SELL ⚠️"
        color = "bearish"
        recommendation = "Sinyal jual kuat. Risiko penurunan tinggi."
    
    return {
        'prediction': prediction,
        'color': color,
        'recommendation': recommendation,
        'signals': signals,
        'score': scores,
        'max_score': max_score,
        'score_percentage': score_percentage
    }

# Fungsi untuk memformat dan menampilkan financial statements
def display_financials(financials_data):
    if financials_data is None:
        st.info("Data financial statements tidak tersedia untuk saham ini")
        return
        
    # 1. BUAT PILIHAN ANNUAL ATAU QUARTAL
    period_type = st.radio(
        "Pilih Periode Laporan:",
        ('Annual', 'Quarterly'),
        horizontal=True
    )
    
    st.markdown("---") # Pemisah
        
    try:
        if period_type == 'Annual':
            data_source = financials_data['annual']
        else:
            data_source = financials_data['quarterly']

        tabs = st.tabs(["💰 Income Statement", "📊 Balance Sheet", "💵 Cash Flow"])
        
        # Fungsi helper internal untuk memproses DataFrame
        def format_and_display(df, period_type):
            if df is None or df.empty:
                return False
            
            # Buat salinan agar tidak mengubah data di cache
            df_copy = df.copy()
            
            # 2. BUAT HEADER JADI TAHUN SAJA (JIKA ANNUAL) ATAU TAHUN-Q (JIKA QUARTAL)
            new_cols = []
            for col in df_copy.columns:
                try:
                    timestamp = pd.to_datetime(col)
                    if period_type == 'Annual':
                        new_cols.append(str(timestamp.year)) # Format: "2023"
                    else: # Quarterly
                        new_cols.append(f"{timestamp.year}-Q{timestamp.quarter}") # Format: "2023-Q4"
                except:
                    new_cols.append(col) # Biarkan kolom non-tanggal
            
            df_copy.columns = new_cols
            
            # 3. BUAT ANGKA JADI SIMPLE (T, B, M)
            df_formatted = df_copy.applymap(format_value)
            
            st.dataframe(df_formatted, use_container_width=True)
            return True

        with tabs[0]:
            st.subheader("Laporan Laba Rugi")
            income_stmt = data_source.get('income')
            if not format_and_display(income_stmt, period_type):
                st.info(f"Data {period_type.lower()} laporan laba rugi tidak tersedia")
        
        with tabs[1]:
            st.subheader("Neraca")
            balance_sheet = data_source.get('balance')
            if not format_and_display(balance_sheet, period_type):
                st.info(f"Data {period_type.lower()} neraca tidak tersedia")
        
        with tabs[2]:
            st.subheader("Arus Kas")
            cash_flow = data_source.get('cashflow')
            if not format_and_display(cash_flow, period_type):
                st.info(f"Data {period_type.lower()} arus kas tidak tersedia")
                
    except Exception as e:
        st.error(f"Terjadi error saat memformat data keuangan: {e}")
        st.info("Data financial statements mungkin tidak lengkap atau tidak tersedia untuk saham ini")

# Main aplikasi
if analyze_button or 'last_stock' in st.session_state:
    if analyze_button:
        st.session_state.last_stock = stock_code
    
    try:
        with st.spinner(f"Menganalisis {stock_code}..."):
            
            # --- MODIFIKASI: PANGGIL FUNGSI CACHE ---
            df = get_stock_history(stock_code, period)
            info = get_stock_info(stock_code)
            # --- AKHIR MODIFIKASI ---
            
            if df is None or info is None:
                st.error(f"❌ Data saham {stock_code} tidak ditemukan atau gagal diambil. Pastikan kode benar (contoh: BBCA.JK)")
            
            else:
                df = calculate_all_indicators(df)
                
                # Header Info (info sudah ada)
                st.markdown(f"## 🏢 {info.get('longName', stock_code)}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price * 100) if prev_price != 0 else 0
                    st.metric("Harga", f"Rp {current_price:,.0f}", f"{change_pct:+.2f}%")
                
                with col2:
                    st.metric("High", f"Rp {df['High'].max():,.0f}")
                
                with col3:
                    st.metric("Low", f"Rp {df['Low'].min():,.0f}")
                
                with col4:
                    st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                
                with col5:
                    if 'marketCap' in info and info['marketCap']:
                        st.metric("Market Cap", f"Rp {info['marketCap']/1e9:.1f}B")
                
                # Tabs untuk organisasi
                main_tabs = st.tabs([
                    "📊 Overview",
                    "📈 Technical Analysis", 
                    "💼 Fundamental",
                    "📋 Financials",
                    "🔑 Key Stats",
                    "👥 Holders",
                    "🎯 Prediction"
                ])
                
                # TAB 1: OVERVIEW
                with main_tabs[0]:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📈 Grafik Harga")
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Price'
                        ))
                        
                        if 'MA20' in df.columns:
                            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')))
                        if 'MA50' in df.columns:
                            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='blue')))
                        
                        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("ℹ️ Info Perusahaan")
                        st.write(f"**Sektor:** {info.get('sector', 'N/A')}")
                        st.write(f"**Industri:** {info.get('industry', 'N/A')}")
                        st.write(f"**Karyawan:** {info.get('fullTimeEmployees', 'N/A'):,}" if 'fullTimeEmployees' in info else "**Karyawan:** N/A")
                        st.write(f"**Website:** {info.get('website', 'N/A')}")
                        
                        st.markdown("---")
                        st.subheader("📝 Deskripsi")
                        description = info.get('longBusinessSummary', 'Tidak ada deskripsi')
                        st.write(description[:300] + "..." if len(description) > 300 else description)
                
                # TAB 2: TECHNICAL ANALYSIS
                with main_tabs[1]:
                    st.subheader("📊 Analisis Teknikal Lengkap")
                    
                    # Indikator Grid
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'RSI' in df.columns:
                            rsi = df['RSI'].iloc[-1]
                            st.metric("RSI (14)", f"{rsi:.2f}", 
                                     "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Normal")
                    
                    with col2:
                        if 'ADX' in df.columns:
                            adx = df['ADX'].iloc[-1]
                            st.metric("ADX", f"{adx:.2f}",
                                     "Strong Trend" if adx > 25 else "Weak Trend")
                    
                    with col3:
                        if 'ATR' in df.columns:
                            atr = df['ATR'].iloc[-1]
                            st.metric("ATR", f"{atr:.2f}", "Volatility")
                    
                    with col4:
                        if 'MFI' in df.columns:
                            mfi = df['MFI'].iloc[-1]
                            st.metric("MFI", f"{mfi:.2f}", "Money Flow")
                    
                    # Charts
                    st.markdown("---")
                    
                    # Bollinger Bands
                    st.subheader("📉 Bollinger Bands")
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='white')))
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='Upper', line=dict(color='red', dash='dash')))
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='Middle', line=dict(color='blue')))
                    fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='Lower', line=dict(color='green', dash='dash')))
                    fig_bb.update_layout(height=400)
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # MACD
                        st.subheader("📊 MACD")
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
                        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')))
                        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram'))
                        fig_macd.update_layout(height=300)
                        st.plotly_chart(fig_macd, use_container_width=True)
                    
                    with col2:
                        # Stochastic
                        st.subheader("🎯 Stochastic Oscillator")
                        fig_stoch = go.Figure()
                        fig_stoch.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K', line=dict(color='blue')))
                        fig_stoch.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D', line=dict(color='red')))
                        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
                        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
                        fig_stoch.update_layout(height=300)
                        st.plotly_chart(fig_stoch, use_container_width=True)
                    
                    # Volume & OBV
                    st.subheader("📊 Volume & OBV Analysis")
                    fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                           subplot_titles=('Volume', 'OBV'))
                    
                    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
                    fig_vol.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=1, col=1)
                    fig_vol.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV', line=dict(color='purple')), row=2, col=1)
                    fig_vol.update_layout(height=400)
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                # TAB 3: FUNDAMENTAL
                with main_tabs[2]:
                    st.subheader("💼 Analisis Fundamental")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 📊 Valuasi")
                        if 'trailingPE' in info and info['trailingPE']:
                            st.metric("P/E Ratio", f"{info['trailingPE']:.2f}")
                        if 'forwardPE' in info and info['forwardPE']:
                            st.metric("Forward P/E", f"{info['forwardPE']:.2f}")
                        if 'priceToBook' in info and info['priceToBook']:
                            st.metric("P/B Ratio", f"{info['priceToBook']:.2f}")
                        if 'enterpriseToRevenue' in info and info['enterpriseToRevenue']:
                            st.metric("P/S Ratio", f"{info['enterpriseToRevenue']:.2f}")
                    
                    with col2:
                        st.markdown("### 💰 Profitabilitas")
                        if 'profitMargins' in info and info['profitMargins']:
                            st.metric("Profit Margin", f"{info['profitMargins']*100:.2f}%")
                        if 'returnOnAssets' in info and info['returnOnAssets']:
                            st.metric("ROA", f"{info['returnOnAssets']*100:.2f}%")
                        if 'returnOnEquity' in info and info['returnOnEquity']:
                            st.metric("ROE", f"{info['returnOnEquity']*100:.2f}%")
                        if 'operatingMargins' in info and info['operatingMargins']:
                            st.metric("Operating Margin", f"{info['operatingMargins']*100:.2f}%")
                    
                    with col3:
                        st.markdown("### 📈 Pertumbuhan & Dividen")
                        if 'revenueGrowth' in info and info['revenueGrowth']:
                            st.metric("Revenue Growth", f"{info['revenueGrowth']*100:.2f}%")
                        if 'earningsGrowth' in info and info['earningsGrowth']:
                            st.metric("Earnings Growth", f"{info['earningsGrowth']*100:.2f}%")
                        if 'dividendYield' in info and info['dividendYield']:
                            st.metric("Dividend Yield", f"{info['dividendYield']*100:.2f}%")
                        if 'payoutRatio' in info and info['payoutRatio']:
                            st.metric("Payout Ratio", f"{info['payoutRatio']*100:.2f}%")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 💪 Kesehatan Finansial")
                        if 'currentRatio' in info and info['currentRatio']:
                            st.metric("Current Ratio", f"{info['currentRatio']:.2f}")
                        if 'quickRatio' in info and info['quickRatio']:
                            st.metric("Quick Ratio", f"{info['quickRatio']:.2f}")
                        if 'debtToEquity' in info and info['debtToEquity']:
                            st.metric("Debt to Equity", f"{info['debtToEquity']:.2f}")
                        if 'totalDebt' in info and info['totalDebt']:
                            st.metric("Total Debt", f"Rp {info['totalDebt']/1e9:.2f}B")
                    
                    with col2:
                        st.markdown("### 📊 Earnings & Revenue")
                        if 'totalRevenue' in info and info['totalRevenue']:
                            st.metric("Total Revenue", f"Rp {info['totalRevenue']/1e9:.2f}B")
                        if 'ebitda' in info and info['ebitda']:
                            st.metric("EBITDA", f"Rp {info['ebitda']/1e9:.2f}B")
                        if 'netIncomeToCommon' in info and info['netIncomeToCommon']:
                            st.metric("Net Income", f"Rp {info['netIncomeToCommon']/1e9:.2f}B")
                        if 'earningsPerShare' in info and info['earningsPerShare']:
                            st.metric("EPS", f"Rp {info['earningsPerShare']:.2f}")
                
                # TAB 4: FINANCIALS
                with main_tabs[3]:
                    financials_data = get_financial_data(stock_code)
                    display_financials(financials_data) # Panggil fungsi yang sudah dimodifikasi
                
                # TAB 5: KEY STATISTICS
                with main_tabs[4]:
                    st.subheader("🔑 Key Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Trading Info")
                        stats_data = {
                            "Beta": info.get('beta', 'N/A'),
                            "52 Week High": f"Rp {info.get('fiftyTwoWeekHigh', 0):,.0f}" if 'fiftyTwoWeekHigh' in info else 'N/A',
                            "52 Week Low": f"Rp {info.get('fiftyTwoWeekLow', 0):,.0f}" if 'fiftyTwoWeekLow' in info else 'N/A',
                            "50 Day Avg": f"Rp {info.get('fiftyDayAverage', 0):,.0f}" if 'fiftyDayAverage' in info else 'N/A',
                            "200 Day Avg": f"Rp {info.get('twoHundredDayAverage', 0):,.0f}" if 'twoHundredDayAverage' in info else 'N/A',
                            "Avg Volume": f"{info.get('averageVolume', 0):,.0f}" if 'averageVolume' in info else 'N/A',
                            "Avg Volume (10d)": f"{info.get('averageVolume10days', 0):,.0f}" if 'averageVolume10days' in info else 'N/A',
                        }
                        for key, value in stats_data.items():
                            st.write(f"**{key}:** {value}")
                    
                    with col2:
                        st.markdown("### 💼 Share Statistics")
                        share_stats = {
                            "Shares Outstanding": f"{info.get('sharesOutstanding', 0):,.0f}" if 'sharesOutstanding' in info else 'N/A',
                            "Float Shares": f"{info.get('floatShares', 0):,.0f}" if 'floatShares' in info else 'N/A',
                            "% Held by Insiders": f"{info.get('heldPercentInsiders', 0)*100:.2f}%" if 'heldPercentInsiders' in info else 'N/A',
                            "% Held by Institutions": f"{info.get('heldPercentInstitutions', 0)*100:.2f}%" if 'heldPercentInstitutions' in info else 'N/A',
                            "Short Ratio": info.get('shortRatio', 'N/A'),
                            "Short % of Float": f"{info.get('shortPercentOfFloat', 0)*100:.2f}%" if 'shortPercentOfFloat' in info else 'N/A',
                        }
                        for key, value in share_stats.items():
                            st.write(f"**{key}:** {value}")
                
                # TAB 6: HOLDERS
                with main_tabs[5]:
                    st.subheader("👥 Holders & Recommendations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🏢 Major Holders")
                        holders_data = get_holders_data(stock_code)
                        if holders_data and not holders_data['major'].empty:
                            st.dataframe(holders_data['major'], use_container_width=True)
                        else:
                            st.info("Data major holders tidak tersedia")
                        
                        st.markdown("### 🏦 Institutional Holders")
                        if holders_data and not holders_data['institutional'].empty:
                            st.dataframe(holders_data['institutional'].head(10), use_container_width=True)
                        else:
                            st.info("Data institutional holders tidak tersedia")
                    
                    with col2:
                        st.markdown("### 📊 Analyst Recommendations")
                        recommendations = get_recommendations_data(stock_code)
                        if recommendations is not None and not recommendations.empty:
                            st.dataframe(recommendations.tail(10), use_container_width=True)
                        else:
                            st.info("Data rekomendasi analis tidak tersedia")
                        
                        st.markdown("### 🎯 Target Price")
                        if 'targetMeanPrice' in info and info['targetMeanPrice']:
                            st.metric("Target Mean", f"Rp {info['targetMeanPrice']:,.0f}")
                        if 'targetHighPrice' in info and info['targetHighPrice']:
                            st.metric("Target High", f"Rp {info['targetHighPrice']:,.0f}")
                        if 'targetLowPrice' in info and info['targetLowPrice']:
                            st.metric("Target Low", f"Rp {info['targetLowPrice']:,.0f}")
                        if 'recommendationKey' in info:
                            st.metric("Recommendation", info['recommendationKey'].upper())
                
                # TAB 7: PREDICTION
                with main_tabs[6]:
                    st.subheader("🎯 Prediksi & Analisis Mendalam")
                    
                    prediction_result = simple_prediction(df, info)
                    
                    # Display Prediction
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div class='insight-box {prediction_result['color']}'>
                            <h2 style='text-align: center; margin: 0;'>{prediction_result['prediction']}</h2>
                            <p style='text-align: center; margin: 10px 0;'>{prediction_result['recommendation']}</p>
                            <p style='text-align: center; margin: 0;'>Skor: {prediction_result['score']}/{prediction_result['max_score']} 
                            ({prediction_result['score_percentage']:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Signal Details
                    st.markdown("### 📊 Detail Sinyal Teknikal")
                    
                    col1, col2 = st.columns(2)
                    
                    for i, (title, desc, score) in enumerate(prediction_result['signals']):
                        if i % 2 == 0:
                            with col1:
                                color = "🟢" if score > 0 else "🔴" if score < 0 else "🟡"
                                st.markdown(f"{color} **{title}**")
                                st.write(f"_{desc}_")
                                st.write("")
                        else:
                            with col2:
                                color = "🟢" if score > 0 else "🔴" if score < 0 else "🟡"
                                st.markdown(f"{color} **{title}**")
                                st.write(f"_{desc}_")
                                st.write("")
                    
                    st.markdown("---")
                    
                    # Support & Resistance
                    st.markdown("### 🎯 Support & Resistance Levels")
                    
                    recent_high = df['High'].tail(20).max()
                    recent_low = df['Low'].tail(20).min()
                    pivot = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
                    
                    r1 = 2 * pivot - recent_low
                    s1 = 2 * pivot - recent_high
                    r2 = pivot + (recent_high - recent_low)
                    s2 = pivot - (recent_high - recent_low)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Resistance 2", f"Rp {r2:,.0f}")
                        st.metric("Resistance 1", f"Rp {r1:,.0f}")
                    with col2:
                        st.metric("Pivot", f"Rp {pivot:,.0f}", "Current Level")
                    with col3:
                        st.metric("Support 1", f"Rp {s1:,.0f}")
                        st.metric("Support 2", f"Rp {s2:,.0f}")
                    
                    st.markdown("---")
                    
                    # Risk Assessment
                    st.markdown("### ⚠️ Risk Assessment")
                    
                    if 'beta' in info and info['beta']:
                        beta = info['beta']
                        if beta > 1.5:
                            risk_level = "TINGGI"
                            risk_color = "🔴"
                            risk_desc = "Volatilitas tinggi, cocok untuk trader agresif"
                        elif beta > 1:
                            risk_level = "SEDANG-TINGGI"
                            risk_color = "🟠"
                            risk_desc = "Volatilitas cukup tinggi"
                        elif beta > 0.5:
                            risk_level = "SEDANG"
                            risk_color = "🟡"
                            risk_desc = "Volatilitas moderat"
                        else:
                            risk_level = "RENDAH"
                            risk_color = "🟢"
                            risk_desc = "Volatilitas rendah, cocok untuk investor konservatif"
                        
                        st.markdown(f"{risk_color} **Risk Level: {risk_level}** (Beta: {beta:.2f})")
                        st.write(risk_desc)
                    
                    if 'ATR' in df.columns:
                        atr_pct = (df['ATR'].iloc[-1] / current_price) * 100
                        st.write(f"📊 **Average True Range:** Rp {df['ATR'].iloc[-1]:,.0f} ({atr_pct:.2f}% dari harga)")
                    
                    st.markdown("---")
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **DISCLAIMER PENTING:**
                    
                    Prediksi ini adalah analisis SEDERHANA berdasarkan indikator teknikal dan tidak menjamin hasil investasi. 
                    Faktor-faktor yang perlu dipertimbangkan:
                    - Kondisi pasar global dan domestik
                    - Berita fundamental perusahaan
                    - Sentimen pasar
                    - Kondisi ekonomi makro
                    - Risiko spesifik industri
                    
                    **Selalu lakukan riset mendalam dan konsultasi dengan advisor keuangan profesional sebelum mengambil keputusan investasi!**
                    """)
                
    except Exception as e:
        st.error(f"❌ Terjadi error: {str(e)}")
        st.info("💡 Coba refresh halaman. Jika masalah berlanjut, kemungkinan API yfinance sedang diblokir.")

else:
    # Welcome screen
    st.info("👈 Masukkan kode saham dan klik **Analisis Saham**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Fitur Lengkap:
        - ✅ **Technical Analysis** - 15+ indikator teknikal
        - ✅ **Fundamental Analysis** - Valuasi, profitabilitas, kesehatan finansial
        - ✅ **Financial Statements** - Income Statement, Balance Sheet, Cash Flow
        - ✅ **Key Statistics** - Data lengkap statistik saham
        - ✅ **Holders Info** - Major & institutional holders
        - ✅ **Smart Prediction** - Prediksi berdasarkan multiple indicators
        - ✅ **Support/Resistance** - Level-level penting
        - ✅ **Risk Assessment** - Analisis risiko investasi
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Indikator Teknikal:
        - Moving Averages (5, 10, 20, 50, 100, 200)
        - Bollinger Bands
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Stochastic Oscillator
        - ADX (Average Directional Index)
        - ATR (Average True Range)
        - OBV (On-Balance Volume)
        - Williams %R
        - CCI, MFI, dan lainnya
        """)
    
    st.markdown("### 💡 Contoh Saham IHSG Populer:")
    
    examples = pd.DataFrame({
        'Kode': ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK', 'UNVR.JK', 'GOTO.JK', 'BREN.JK'],
        'Nama': ['Bank BCA', 'Bank BRI', 'Bank Mandiri', 'Telkom', 'Astra International', 'Unilever', 'GoTo', 'Barito Renewables'],
        'Sektor': ['Banking', 'Banking', 'Banking', 'Telco', 'Automotive', 'Consumer Goods', 'Technology', 'Energy']
    })
    
    st.dataframe(examples, use_container_width=True, hide_index=True)