"""
Streamlit Dashboard - 实时版（Binance 数据）
================================
使用 Binance 公开 API，获取真正的实时价格数据

特点:
- ✅ 实时价格（秒级更新）
- ✅ 1分钟/5分钟 K线图
- ✅ 自动刷新（可配置间隔）
- ✅ 完全免费，无需 API Key
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

from src.data_collection.binance_public import BinancePublicAPI

st.set_page_config(
    page_title="BTC实时价格监控",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 样式
st.markdown("""
<style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
    }
    .price-up {
        color: #00ff00;
        font-size: 40px;
        font-weight: bold;
    }
    .price-down {
        color: #ff0000;
        font-size: 40px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_api():
    """获取 Binance API 客户端（缓存）"""
    return BinancePublicAPI()


def fetch_realtime_data_sync():
    """获取实时数据（同步版本）"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_fetch_realtime_data())
        loop.close()
        return result
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None, None


async def _fetch_realtime_data():
    """内部异步获取实时数据"""
    api = get_api()
    
    # 并行获取多个数据
    price_task = api.get_current_price("BTCUSDT")
    ticker_task = api.get_ticker_24h("BTCUSDT")
    
    price_data, ticker_data = await asyncio.gather(price_task, ticker_task)
    
    return price_data, ticker_data


def fetch_klines_sync(interval: str, days: int):
    """获取 K 线数据（同步版本）"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        api = get_api()
        result = loop.run_until_complete(api.get_klines("BTCUSDT", interval, days))
        loop.close()
        return result
    except Exception as e:
        st.error(f"获取K线数据失败: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    if len(df) < 20:
        return df
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 移动平均线
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # 布林带
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    return df


def create_price_chart(df: pd.DataFrame):
    """创建价格图表"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('价格走势', 'RSI')
    )
    
    # 蜡烛图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='价格'
        ),
        row=1, col=1
    )
    
    # 移动平均线
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # 布林带
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                      line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                      line=dict(color='gray', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # RSI 超买超卖线
        # Note: Plotly's add_hline expects string for row/col in subplots
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title="BTC/USDT 实时走势",
        xaxis_title="时间",
        yaxis_title="价格 (USDT)",
        height=800,
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def main():
    st.title("📈 BTC/USDT 实时价格监控")
    st.markdown("*数据来源: Binance 公开 API (免费)*")
    
    # 侧边栏设置
    st.sidebar.header("⚙️ 设置")
    
    # 刷新间隔
    refresh_interval = st.sidebar.selectbox(
        "自动刷新间隔",
        options=[5, 10, 15, 30, 60],
        index=2,
        format_func=lambda x: f"{x} 秒"
    )
    
    # K线周期
    kline_interval = st.sidebar.selectbox(
        "K线周期",
        options=["1m", "5m", "15m", "1h", "4h"],
        index=1,
        format_func=lambda x: {
            "1m": "1分钟", "5m": "5分钟", "15m": "15分钟",
            "1h": "1小时", "4h": "4小时"
        }[x]
    )
    
    # 历史天数
    history_days = st.sidebar.slider(
        "历史数据天数",
        min_value=1,
        max_value=30,
        value=7
    )
    
    # 手动刷新按钮
    if st.sidebar.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()
    
    # 获取实时数据
    try:
        price_data, ticker_data = fetch_realtime_data_sync()
        
        if ticker_data is None or price_data is None:
            st.error("无法获取实时数据，请稍后重试")
            return
        
        # 显示实时价格
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_class = "price-up" if ticker_data['change'] >= 0 else "price-down"
            st.markdown(f"### 当前价格")
            st.markdown(f"<p class='{price_class}'>${ticker_data['price']:,.2f}</p>", 
                       unsafe_allow_html=True)
        
        with col2:
            change_emoji = "📈" if ticker_data['change'] >= 0 else "📉"
            st.metric(
                "24h 涨跌",
                f"${ticker_data['change']:+,.2f}",
                f"{ticker_data['change_percent']:+.2f}%"
            )
        
        with col3:
            st.metric(
                "24h 最高",
                f"${ticker_data['high']:,.2f}"
            )
        
        with col4:
            st.metric(
                "24h 最低",
                f"${ticker_data['low']:,.2f}"
            )
        
        # 显示成交量
        col5, col6 = st.columns(2)
        with col5:
            st.metric("24h 成交量 (BTC)", f"{ticker_data['volume']:,.2f}")
        with col6:
            st.metric("24h 成交额 (USDT)", f"${ticker_data['quote_volume']:,.0f}")
        
        # 获取 K 线数据
        st.markdown("---")
        st.subheader("📊 价格走势图")
        
        with st.spinner("正在加载K线数据..."):
            df = fetch_klines_sync(kline_interval, history_days)
            
            if not df.empty:
                # 计算技术指标
                df = calculate_technical_indicators(df)
                
                # 显示图表
                fig = create_price_chart(df)
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示数据表格
                with st.expander("📋 查看原始数据"):
                    st.dataframe(df.tail(50))
            else:
                st.error("无法获取K线数据")
        
        # 显示更新时间
        st.sidebar.markdown("---")
        st.sidebar.info(f"🕐 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 自动刷新
        if refresh_interval:
            import time
            time.sleep(refresh_interval)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ 获取数据失败: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
