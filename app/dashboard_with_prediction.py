"""
Streamlit Dashboard - 带预测功能版本
================================
实时监控 + AI 预测 + 技术分析
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
import torch
import nest_asyncio

# 允许嵌套事件循环
nest_asyncio.apply()

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TradingConfig, ModelConfig
from src.data_collection.coingecko_fetcher import CoinGeckoFetcher
from src.features.engineer import FeatureEngineer
from src.models.gru import GRUPredictor
from src.models.lightgbm_model import LightGBMPredictor

st.set_page_config(
    page_title="BTC趋势预测系统 - AI版",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 样式
st.markdown("""
<style>
    .big-font {
        font-size: 40px !important;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .pred-bullish {
        background: linear-gradient(135deg, #00ff00 0%, #00cc00 100%);
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .pred-bearish {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .pred-neutral {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_type: str):
    """加载训练好的模型"""
    try:
        model_dir = Path(__file__).parent.parent / "models" / "saved"
        
        if model_type == "GRU":
            model_path = model_dir / "gru_best.pth"
            if not model_path.exists():
                return None
            
            # 加载检查点以获取输入形状
            checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            
            # 创建模型
            model = GRUPredictor(
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # 从检查点获取输入形状
            if 'config' in checkpoint and 'input_shape' in checkpoint['config']:
                input_shape = checkpoint['config']['input_shape']
            else:
                input_shape = (24, 100)
            
            model.build(input_shape=input_shape, n_classes=3)
            model.load(model_path)
            
        elif model_type == "LightGBM":
            model_path = model_dir / "lightgbm_best.txt"
            if not model_path.exists():
                return None
            model = LightGBMPredictor()
            model.load(model_path)
        else:
            return None
        
        return model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


@st.cache_resource
def get_feature_engineer():
    """获取特征工程器"""
    return FeatureEngineer()


def fetch_data_with_features(days: int = 7):
    """获取数据并生成特征"""
    try:
        # 获取原始数据
        fetcher = CoinGeckoFetcher()
        
        async def get_data():
            return await fetcher.get_ohlc("bitcoin", days=days)
        
        ohlc_list = asyncio.run(get_data())
        
        if not ohlc_list:
            return None, None
        
        # 转换为 DataFrame
        df = pd.DataFrame([{
            'timestamp': ohlc.timestamp,
            'open': ohlc.open,
            'high': ohlc.high,
            'low': ohlc.low,
            'close': ohlc.close,
            'volume': ohlc.volume
        } for ohlc in ohlc_list])
        df = df.set_index('timestamp')
        
        # 生成特征
        engineer = get_feature_engineer()
        df_features = engineer.create_features(df)
        
        return df, df_features
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None, None


def make_prediction(model, df_features):
    """使用模型进行预测"""
    try:
        if model is None or df_features is None or df_features.empty:
            return None, None
        
        # 准备最近的数据
        window_size = 24  # 使用最近24小时数据
        if len(df_features) < window_size:
            return None, None
        
        # 获取最近的特征
        recent_data = df_features.iloc[-window_size:].values
        
        # 标准化（简单版本）
        mean = recent_data.mean(axis=0)
        std = recent_data.std(axis=0) + 1e-8
        X = (recent_data - mean) / std
        
        # 为GRU重塑形状 (1, window_size, features)
        X = X.reshape(1, window_size, -1)
        
        # 预测
        pred_proba = model.predict_proba(X)
        pred_class = model.predict(X)
        
        return pred_class[0], pred_proba[0]
    except Exception as e:
        st.error(f"预测失败: {e}")
        return None, None


def main():
    st.title("🤖 BTC趋势预测系统 - AI增强版")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 模型选择
        model_type = st.selectbox(
            "选择预测模型",
            ["GRU", "LightGBM", "无（仅显示数据）"],
            index=0
        )
        
        # 数据范围
        days = st.slider("历史数据天数", 1, 30, 7)
        
        # 刷新按钮
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.info("💡 提示: 模型需要先训练才能使用")
    
    # 加载模型
    model = None
    if model_type != "无（仅显示数据）":
        with st.spinner(f"加载 {model_type} 模型..."):
            model = load_model(model_type)
            if model is None:
                st.warning(f"⚠️ {model_type} 模型未找到，请先运行训练")
    
    # 获取数据
    with st.spinner("获取市场数据..."):
        df_raw, df_features = fetch_data_with_features(days)
    
    if df_raw is None or df_raw.empty:
        st.error("❌ 无法获取数据，请检查网络连接")
        return
    
    # 显示当前价格
    current_price = df_raw['close'].iloc[-1]
    price_change = df_raw['close'].pct_change().iloc[-1] * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "当前价格",
            f"${current_price:,.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        st.metric("24h 最高", f"${df_raw['high'].iloc[-24:].max():,.2f}")
    
    with col3:
        st.metric("24h 最低", f"${df_raw['low'].iloc[-24:].min():,.2f}")
    
    with col4:
        volume_24h = df_raw['volume'].iloc[-24:].sum()
        st.metric("24h 成交量", f"${volume_24h/1e9:.2f}B")
    
    # AI 预测区域
    if model is not None:
        st.markdown("---")
        st.header("🎯 AI 预测")
        
        pred_class, pred_proba = make_prediction(model, df_features)
        
        if pred_class is not None and pred_proba is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # 预测结果
                labels = ["看跌 📉", "震荡 ➡️", "看涨 📈"]
                colors = ["pred-bearish", "pred-neutral", "pred-bullish"]
                
                st.markdown(
                    f'<div class="prediction-box {colors[pred_class]}">'
                    f'{labels[pred_class]}<br>'
                    f'置信度: {pred_proba[pred_class]*100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # 建议
                if pred_class == 2:  # 看涨
                    st.success("💡 建议: 考虑买入或持有")
                elif pred_class == 0:  # 看跌
                    st.error("💡 建议: 考虑卖出或观望")
                else:  # 震荡
                    st.warning("💡 建议: 保持观望，等待明确信号")
            
            with col2:
                # 概率分布图
                fig_prob = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=pred_proba * 100,
                        marker=dict(
                            color=['#ff4444', '#ffaa00', '#44ff44'],
                            line=dict(color='white', width=2)
                        ),
                        text=[f'{p*100:.1f}%' for p in pred_proba],
                        textposition='auto',
                    )
                ])
                
                fig_prob.update_layout(
                    title="预测概率分布",
                    xaxis_title="趋势方向",
                    yaxis_title="概率 (%)",
                    height=300,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.warning("⚠️ 数据不足，无法进行预测（需要至少24小时数据）")
    
    # 价格图表
    st.markdown("---")
    st.header("📊 价格走势")
    
    # 创建K线图
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('BTC/USDT 价格', '成交量')
    )
    
    # K线
    fig.add_trace(
        go.Candlestick(
            x=df_raw.index,
            open=df_raw['open'],
            high=df_raw['high'],
            low=df_raw['low'],
            close=df_raw['close'],
            name='OHLC',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )
    
    # 移动平均线
    ma_periods = [7, 25, 99]
    ma_colors = ['#ffaa00', '#00aaff', '#ff00ff']
    for period, color in zip(ma_periods, ma_colors):
        if len(df_raw) >= period:
            ma = df_raw['close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df_raw.index,
                    y=ma,
                    name=f'MA{period}',
                    line=dict(color=color, width=1.5)
                ),
                row=1, col=1
            )
    
    # 成交量
    colors = ['#ff0000' if df_raw['close'].iloc[i] < df_raw['open'].iloc[i] else '#00ff00' 
              for i in range(len(df_raw))]
    
    fig.add_trace(
        go.Bar(
            x=df_raw.index,
            y=df_raw['volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="价格 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 技术指标
    if df_features is not None and not df_features.empty:
        st.markdown("---")
        st.header("📈 技术指标")
        
        tab1, tab2, tab3 = st.tabs(["趋势指标", "动量指标", "成交量指标"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI
                if 'rsi' in df_features.columns:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['rsi'],
                        name='RSI',
                        line=dict(color='#00ffff', width=2)
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
                    fig_rsi.update_layout(
                        title="RSI 相对强弱指标",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # MACD
                if all(col in df_features.columns for col in ['macd', 'macd_signal']):
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['macd'],
                        name='MACD',
                        line=dict(color='#00ff00', width=2)
                    ))
                    fig_macd.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['macd_signal'],
                        name='Signal',
                        line=dict(color='#ff0000', width=2)
                    ))
                    if 'macd_diff' in df_features.columns:
                        colors = ['green' if x > 0 else 'red' for x in df_features['macd_diff']]
                        fig_macd.add_trace(go.Bar(
                            x=df_features.index,
                            y=df_features['macd_diff'],
                            name='Histogram',
                            marker_color=colors,
                            opacity=0.5
                        ))
                    fig_macd.update_layout(
                        title="MACD 指标",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # 布林带
                if all(col in df_features.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    fig_bb = go.Figure()
                    fig_bb.add_trace(go.Scatter(
                        x=df_raw.index,
                        y=df_raw['close'],
                        name='价格',
                        line=dict(color='white', width=2)
                    ))
                    fig_bb.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['bb_upper'],
                        name='上轨',
                        line=dict(color='red', width=1, dash='dash')
                    ))
                    fig_bb.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['bb_middle'],
                        name='中轨',
                        line=dict(color='yellow', width=1)
                    ))
                    fig_bb.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['bb_lower'],
                        name='下轨',
                        line=dict(color='green', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(100,100,100,0.2)'
                    ))
                    fig_bb.update_layout(
                        title="布林带",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_bb, use_container_width=True)
            
            with col2:
                # ATR
                if 'atr' in df_features.columns:
                    fig_atr = go.Figure()
                    fig_atr.add_trace(go.Scatter(
                        x=df_features.index,
                        y=df_features['atr'],
                        name='ATR',
                        line=dict(color='#ff00ff', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255,0,255,0.2)'
                    ))
                    fig_atr.update_layout(
                        title="ATR 平均真实波幅",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_atr, use_container_width=True)
        
        with tab3:
            # 成交量分析
            if 'obv' in df_features.columns:
                fig_obv = go.Figure()
                fig_obv.add_trace(go.Scatter(
                    x=df_features.index,
                    y=df_features['obv'],
                    name='OBV',
                    line=dict(color='#ffaa00', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255,170,0,0.2)'
                ))
                fig_obv.update_layout(
                    title="OBV 能量潮指标",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig_obv, use_container_width=True)
    
    # 页脚
    st.markdown("---")
    st.caption(f"📅 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("⚠️ 免责声明: 本系统仅供参考，不构成投资建议")


if __name__ == "__main__":
    main()
