"""
Streamlit Dashboard - 实时自动刷新版
================================
每15秒自动刷新价格和预测信号
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TradingConfig, ModelConfig
from src.data_collection.coingecko_fetcher import CoinGeckoFetcher

st.set_page_config(
    page_title="BTC趋势预测系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
    }
    .signal-buy {
        color: #00ff00;
        font-size: 30px;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff0000;
        font-size: 30px;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffff00;
        font-size: 30px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
    }
    .refresh-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0,255,0,0.3);
        padding: 5px 10px;
        border-radius: 5px;
        z-index: 9999;
    }
</style>
""", unsafe_allow_html=True)


async def fetch_real_btc_data(days: int = 7) -> pd.DataFrame:
    """从CoinGecko获取真实BTC数据"""
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
            # 计算技术指标
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
        st.error(f"获取数据失败: {e}")
    
    return pd.DataFrame()


def calculate_signal(df: pd.DataFrame) -> tuple:
    """根据技术指标计算交易信号"""
    if df.empty or len(df) < 2:
        return 'HOLD', 0.5
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 初始化信号分数
    score = 0
    factors = 0
    
    # RSI 信号
    if 'rsi' in df.columns and not pd.isna(latest['rsi']):
        factors += 1
        if latest['rsi'] < 30:
            score += 1  # 超卖，买入
        elif latest['rsi'] > 70:
            score -= 1  # 超买，卖出
    
    # SMA 交叉信号
    if 'sma_24' in df.columns and 'sma_72' in df.columns:
        if not pd.isna(latest['sma_24']) and not pd.isna(latest['sma_72']):
            factors += 1
            if latest['sma_24'] > latest['sma_72']:
                score += 1  # 金叉，买入
            else:
                score -= 1  # 死叉，卖出
    
    # 布林带信号
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        if not pd.isna(latest['bb_upper']) and not pd.isna(latest['bb_lower']):
            factors += 1
            bb_range = latest['bb_upper'] - latest['bb_lower']
            if bb_range > 0:
                bb_position = (latest['close'] - latest['bb_lower']) / bb_range
                if bb_position < 0.2:
                    score += 1  # 接近下轨，买入
                elif bb_position > 0.8:
                    score -= 1  # 接近上轨，卖出
    
    # 价格动量
    factors += 1
    price_change = (latest['close'] - prev['close']) / prev['close']
    if price_change > 0.01:
        score += 1
    elif price_change < -0.01:
        score -= 1
    
    # 计算信号和置信度
    if factors > 0:
        avg_score = score / factors
        confidence = min(abs(avg_score) * 0.8 + 0.2, 0.95)
        
        if avg_score > 0.3:
            return 'BUY', confidence
        elif avg_score < -0.3:
            return 'SELL', confidence
        else:
            return 'HOLD', max(confidence * 0.6, 0.5)
    
    return 'HOLD', 0.5


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


def create_sentiment_gauge(sentiment_score: float) -> go.Figure:
    """创建情感得分仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "市场情感"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "red"},
                {'range': [-0.3, 0.3], 'color': "gray"},
                {'range': [0.3, 1], 'color': "green"}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,
        template='plotly_dark'
    )
    
    return fig


def main():
    """主函数"""
    st.title("🚀 BTC趋势预测系统")
    
    # 初始化 session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = 0
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 15  # 默认15秒
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        st.subheader("自动刷新")
        auto_refresh_enabled = st.checkbox("启用自动刷新", value=True)
        
        if auto_refresh_enabled:
            refresh_interval = st.slider(
                "刷新间隔（秒）",
                min_value=5,
                max_value=60,
                value=15,
                step=5
            )
            st.session_state.refresh_interval = refresh_interval
        
        st.subheader("显示选项")
        show_technical = st.checkbox("显示技术指标", value=True)
        show_sentiment = st.checkbox("显示情感分析", value=True)
        
        st.markdown("---")
        st.subheader("模型状态")
        model_status = st.empty()
        
        st.markdown("---")
        st.subheader("最后更新")
        last_update_display = st.empty()
    
    st.markdown("---")
    
    # 主内容区
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
    
    # 图表区域
    col_chart, col_info = st.columns([3, 1])
    
    with col_chart:
        chart_placeholder = st.empty()
    
    with col_info:
        if show_sentiment:
            st.subheader("💭 市场情感")
            sentiment_placeholder = st.empty()
        
        if show_technical:
            st.subheader("📊 技术指标")
            tech_indicators = st.empty()
    
    # 检查是否需要刷新
    current_time = time.time()
    should_refresh = (current_time - st.session_state.last_update) >= st.session_state.refresh_interval
    
    if should_refresh or st.session_state.last_update == 0:
        st.session_state.last_update = current_time
        
        # 获取数据
        with st.spinner("正在获取最新数据..."):
            df = asyncio.run(fetch_real_btc_data(days=7))
        
        if not df.empty:
            # 更新显示
            current_price = df['close'].iloc[-1]
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100
            
            price_placeholder.metric(
                "BTC/USDT",
                f"${current_price:,.2f}",
                f"{price_change:+.2f}%"
            )
            
            # 计算信号
            signal, confidence = calculate_signal(df)
            
            if signal == 'BUY':
                signal_placeholder.markdown('<p class="signal-buy">📈 买入</p>', unsafe_allow_html=True)
            elif signal == 'SELL':
                signal_placeholder.markdown('<p class="signal-sell">📉 卖出</p>', unsafe_allow_html=True)
            else:
                signal_placeholder.markdown('<p class="signal-hold">⏸️ 观望</p>', unsafe_allow_html=True)
            
            confidence_placeholder.progress(confidence)
            confidence_placeholder.text(f"{confidence:.1%}")
            
            # 图表
            chart_placeholder.plotly_chart(
                create_price_chart(df.dropna()), 
                use_container_width=True
            )
            
            # 技术指标
            if show_technical and 'rsi' in df.columns:
                latest = df.iloc[-1]
                tech_data = {
                    "RSI": f"{latest['rsi']:.1f}" if not pd.isna(latest['rsi']) else "N/A",
                    "SMA 24": f"${latest['sma_24']:.2f}" if not pd.isna(latest['sma_24']) else "N/A",
                    "SMA 72": f"${latest['sma_72']:.2f}" if not pd.isna(latest['sma_72']) else "N/A",
                    "布林带上轨": f"${latest['bb_upper']:.2f}" if not pd.isna(latest['bb_upper']) else "N/A",
                    "布林带下轨": f"${latest['bb_lower']:.2f}" if not pd.isna(latest['bb_lower']) else "N/A"
                }
                tech_indicators.json(tech_data)
            
            # 情感（基于技术指标计算）
            if show_sentiment:
                sentiment_score = 0
                if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]):
                    rsi = df['rsi'].iloc[-1]
                    if rsi < 30:
                        sentiment_score = -0.5
                    elif rsi > 70:
                        sentiment_score = 0.5
                    else:
                        sentiment_score = (rsi - 50) / 100
                
                sentiment_placeholder.plotly_chart(
                    create_sentiment_gauge(sentiment_score),
                    use_container_width=True
                )
            
            model_status.success("✅ 使用 CoinGecko 实时数据")
            last_update_display.info(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        
        else:
            st.error("无法获取数据，请检查网络连接")
            model_status.error("❌ 数据加载失败")
    
    else:
        # 显示倒计时
        remaining = st.session_state.refresh_interval - (current_time - st.session_state.last_update)
        last_update_display.info(f"下次刷新: {remaining:.0f}秒")
    
    # 自动刷新机制
    if auto_refresh_enabled:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
