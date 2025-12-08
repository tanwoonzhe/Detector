"""
Streamlit Dashboard - Fixed Version
================================
显示真实BTC数据，不调用完整特征工程避免清空
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TradingConfig, ModelConfig
from src.data_collection.coingecko_fetcher import CoinGeckoFetcher

st.set_page_config(
    page_title="BTC趋势预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .signal-buy { color: #00ff00; font-size: 30px; font-weight: bold; }
    .signal-sell { color: #ff0000; font-size: 30px; font-weight: bold; }
    .signal-hold { color: #ffff00; font-size: 30px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


async def fetch_real_btc_data(days: int = 7) -> pd.DataFrame:
    """从CoinGecko获取真实BTC数据，简单计算技术指标"""
    try:
        fetcher = CoinGeckoFetcher()
        market_data = await fetcher.get_hourly_ohlcv(
            symbol="bitcoin",
            days=days,
            vs_currency="usd"
        )
        await fetcher.close()
        
        df = market_data.to_dataframe()
        
        if not df.empty:
            # 简单计算RSI/SMA/布林带，不调用完整特征工程
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['sma_24'] = df['close'].rolling(24).mean()
            df['sma_72'] = df['close'].rolling(72).mean()
            
            df['bb_middle'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            return df
    except Exception as e:
        print(f"获取真实数据失败: {e}")
    
    return pd.DataFrame()


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """创建K线图"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('BTC/USDT K线', '成交量', 'RSI')
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    if 'sma_24' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_24'], 
                      name='SMA 24', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'sma_72' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_72'], 
                      name='SMA 72', line=dict(color='purple', width=1)),
            row=1, col=1
        )
    
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], 
                      name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], 
                      name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    colors = ['green' if c >= o else 'red' 
              for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                      line=dict(color='cyan', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title='BTC/USDT 实时行情',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )
    
    return fig


def main():
    st.title("🚀 BTC趋势预测系统")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ 配置")
        auto_refresh = st.checkbox("自动刷新 (30分钟)", value=False)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📊 当前价格")
        price_placeholder = st.empty()
    
    with col2:
        st.subheader("📈 预测信号")
        signal_placeholder = st.empty()
    
    with col3:
        st.subheader("🎯 置信度")
        confidence_placeholder = st.empty()
    
    st.markdown("---")
    chart_placeholder = st.empty()
    
    # 获取真实数据
    try:
        demo_df = asyncio.run(fetch_real_btc_data(days=7))
        if demo_df.empty:
            raise ValueError("CoinGecko返回空数据")
        st.sidebar.success("✓ 已加载真实数据")
    except Exception as e:
        st.sidebar.warning(f"使用模拟数据: {str(e)[:30]}")
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
        base_price = 65000
        returns = np.random.randn(168) * 0.01
        prices = base_price * np.cumprod(1 + returns)
        
        demo_df = pd.DataFrame({
            'open': prices * (1 - np.random.rand(168) * 0.005),
            'high': prices * (1 + np.random.rand(168) * 0.01),
            'low': prices * (1 - np.random.rand(168) * 0.01),
            'close': prices,
            'volume': np.random.rand(168) * 1e9
        }, index=dates)
        
        delta = demo_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        demo_df['rsi'] = 100 - (100 / (1 + rs))
        demo_df['sma_24'] = demo_df['close'].rolling(24).mean()
        demo_df['sma_72'] = demo_df['close'].rolling(72).mean()
        demo_df['bb_middle'] = demo_df['close'].rolling(20).mean()
        demo_df['bb_std'] = demo_df['close'].rolling(20).std()
        demo_df['bb_upper'] = demo_df['bb_middle'] + 2 * demo_df['bb_std']
        demo_df['bb_lower'] = demo_df['bb_middle'] - 2 * demo_df['bb_std']
    
    current_price = demo_df['close'].iloc[-1]
    price_change = (demo_df['close'].iloc[-1] / demo_df['close'].iloc[-2] - 1) * 100
    
    price_placeholder.metric(
        "BTC/USDT",
        f"${current_price:,.2f}",
        f"{price_change:+.2f}%"
    )
    
    demo_signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.3, 0.3])
    demo_confidence = np.random.uniform(0.5, 0.9)
    
    if demo_signal == 'BUY':
        signal_placeholder.markdown('<p class="signal-buy">📈 买入</p>', unsafe_allow_html=True)
    elif demo_signal == 'SELL':
        signal_placeholder.markdown('<p class="signal-sell">📉 卖出</p>', unsafe_allow_html=True)
    else:
        signal_placeholder.markdown('<p class="signal-hold">⏸️ 观望</p>', unsafe_allow_html=True)
    
    confidence_placeholder.progress(demo_confidence)
    confidence_placeholder.text(f"{demo_confidence:.1%}")
    
    chart_placeholder.plotly_chart(
        create_price_chart(demo_df.dropna()), 
        use_container_width=True
    )
    
    if auto_refresh:
        st.rerun()


if __name__ == "__main__":
    main()
