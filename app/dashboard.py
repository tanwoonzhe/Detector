"""
Streamlit Dashboard
================================
BTC趋势预测系统的Web界面

功能:
1. 实时价格显示
2. 预测信号展示
3. 技术指标图表
4. 情感分析展示
5. 历史表现回顾
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

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TradingConfig, ModelConfig

# 页面配置
st.set_page_config(
    page_title="BTC趋势预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
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
</style>
""", unsafe_allow_html=True)


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """创建K线图"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('BTC/USDT K线', '成交量', 'RSI')
    )
    
    # K线图
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
    
    # 移动平均线
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
    
    # 布林带
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
    
    # 成交量
    colors = ['green' if c >= o else 'red' 
              for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', 
                      line=dict(color='cyan', width=1)),
            row=3, col=1
        )
        # RSI参考线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row="3", col="1")
        fig.add_hline(y=30, line_dash="dash", line_color="green", row="3", col="1")
    
    fig.update_layout(
        title='BTC/USDT 实时行情',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )
    
    return fig


def create_prediction_gauge(confidence: float, signal: str) -> go.Figure:
    """创建预测置信度仪表盘"""
    colors = {
        'BUY': 'green',
        'SELL': 'red', 
        'HOLD': 'yellow'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"预测信号: {signal}"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': colors.get(signal, 'gray')},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "gray"},
                {'range': [60, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        template='plotly_dark'
    )
    
    return fig


def create_window_predictions_chart(predictions: dict) -> go.Figure:
    """创建多窗口预测柱状图"""
    windows = list(predictions.keys())
    values = [predictions[w] for w in windows]
    
    colors = ['green' if v == 2 else ('red' if v == 0 else 'yellow') for v in values]
    labels = ['上涨' if v == 2 else ('下跌' if v == 0 else '横盘') for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{w}h" for w in windows],
            y=[1] * len(windows),
            marker_color=colors,
            text=labels,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="各时间窗口预测",
        xaxis_title="预测窗口",
        yaxis_visible=False,
        height=200,
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
    st.markdown("---")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        auto_refresh = st.checkbox("自动刷新 (30分钟)", value=False)
        
        st.subheader("预测窗口")
        selected_windows = st.multiselect(
            "选择预测窗口",
            options=[0.5, 1, 2, 4],
            default=[0.5, 1, 2, 4]
        )
        
        st.subheader("显示选项")
        show_technical = st.checkbox("显示技术指标", value=True)
        show_sentiment = st.checkbox("显示情感分析", value=True)
        show_history = st.checkbox("显示历史表现", value=False)
        
        st.markdown("---")
        st.subheader("模型状态")
        model_status = st.empty()
    
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
        st.subheader("💭 市场情感")
        sentiment_placeholder = st.empty()
        
        st.subheader("⏱️ 各窗口预测")
        windows_placeholder = st.empty()
    
    st.markdown("---")
    
    # 详细信息区域
    if show_technical:
        st.subheader("📉 技术指标详情")
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.metric("RSI", "54.3", "2.1")
        with tech_col2:
            st.metric("MACD", "125.4", "↑")
        with tech_col3:
            st.metric("布林带位置", "0.65", "中性")
        with tech_col4:
            st.metric("ADX", "28.5", "趋势中等")
    
    # 历史表现
    if show_history:
        st.subheader("📜 历史预测表现")
        history_df = pd.DataFrame({
            '日期': pd.date_range(end=datetime.now(), periods=7, freq='D'),
            '预测': ['BUY', 'HOLD', 'SELL', 'BUY', 'BUY', 'HOLD', 'SELL'],
            '实际': ['上涨', '横盘', '下跌', '上涨', '下跌', '横盘', '下跌'],
            '准确': ['✅', '✅', '✅', '✅', '❌', '✅', '✅']
        })
        st.dataframe(history_df, use_container_width=True)
    
    # 演示数据 (实际使用时替换为真实数据)
    demo_mode = st.sidebar.checkbox("演示模式", value=True)
    
    if demo_mode:
        # 生成演示数据
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
        
        # 模拟价格数据
        base_price = 65000
        returns = np.random.randn(168) * 0.01
        prices = base_price * np.cumprod(1 + returns)
        
        demo_df = pd.DataFrame({
            'open': prices * (1 - np.random.rand(168) * 0.005),
            'high': prices * (1 + np.random.rand(168) * 0.01),
            'low': prices * (1 - np.random.rand(168) * 0.01),
            'close': prices,
            'volume': np.random.rand(168) * 1000000000
        }, index=dates)
        
        # 添加技术指标
        demo_df['sma_24'] = demo_df['close'].rolling(24).mean()
        demo_df['sma_72'] = demo_df['close'].rolling(72).mean()
        demo_df['bb_middle'] = demo_df['close'].rolling(20).mean()
        demo_df['bb_std'] = demo_df['close'].rolling(20).std()
        demo_df['bb_upper'] = demo_df['bb_middle'] + 2 * demo_df['bb_std']
        demo_df['bb_lower'] = demo_df['bb_middle'] - 2 * demo_df['bb_std']
        
        # RSI计算
        delta = demo_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        demo_df['rsi'] = 100 - (100 / (1 + rs))
        
        # 更新显示
        current_price = demo_df['close'].iloc[-1]
        price_change = (demo_df['close'].iloc[-1] / demo_df['close'].iloc[-2] - 1) * 100
        
        price_placeholder.metric(
            "BTC/USDT",
            f"${current_price:,.2f}",
            f"{price_change:+.2f}%"
        )
        
        # 模拟信号
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
        
        # 图表
        chart_placeholder.plotly_chart(
            create_price_chart(demo_df.dropna()), 
            use_container_width=True
        )
        
        # 情感
        if show_sentiment:
            demo_sentiment = np.random.uniform(-0.5, 0.5)
            sentiment_placeholder.plotly_chart(
                create_sentiment_gauge(demo_sentiment),
                use_container_width=True
            )
        
        # 各窗口预测
        demo_predictions = {
            0.5: np.random.choice([0, 1, 2]),
            1: np.random.choice([0, 1, 2]),
            2: np.random.choice([0, 1, 2]),
            4: np.random.choice([0, 1, 2])
        }
        windows_placeholder.plotly_chart(
            create_window_predictions_chart(demo_predictions),
            use_container_width=True
        )
        
        model_status.success("模型已加载 ✅")
    
    else:
        st.info("请先训练模型或加载预训练模型")
        model_status.warning("模型未加载 ⚠️")
    
    # 自动刷新
    if auto_refresh:
        st.rerun()


if __name__ == "__main__":
    main()
