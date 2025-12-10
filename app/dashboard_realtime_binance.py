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
from typing import Optional
import asyncio
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.binance_public import BinancePublicAPI
from src.features.engineer import FeatureEngineer
from src.models.gru import GRUPredictor
from src.models.lightgbm_model import LightGBMPredictor
from src.models.bilstm import BiLSTMPredictor
from src.models.model_manager import ModelManager, ModelInfo

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
</style>
""", unsafe_allow_html=True)


# 不要缓存 API 实例，因为它包含 aiohttp session 会绑定到特定的事件循环


@st.cache_resource
def load_model(model_type: str):
    """加载训练好的模型"""
    try:
        model_dir = Path(__file__).parent.parent / "models" / "saved"
        
        if model_type == "GRU":
            model_path = model_dir / "gru_best.pth"
            if not model_path.exists():
                return None
            
            # 创建模型
            model = GRUPredictor(
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # 使用auto_build自动从checkpoint读取配置并构建模型
            model.load(model_path, auto_build=True)
            
        elif model_type == "BiLSTM":
            model_path = model_dir / "bilstm_best.pth"
            if not model_path.exists():
                return None
            
            model = BiLSTMPredictor(
                hidden_size=128,
                num_layers=2,
                dropout=0.3,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            model.load(model_path, auto_build=True)
            
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


def make_prediction(model, df):
    """使用模型进行预测"""
    try:
        if model is None or df is None or df.empty:
            return None, None
        
        # 生成特征
        engineer = get_feature_engineer()
        df_features = engineer.create_features(df)
        
        if df_features.empty:
            return None, None
        
        # 获取特征列
        feature_cols = engineer.get_feature_columns(df_features)
        n_raw_features = len(feature_cols)
        
        # 检查模型期望的特征数
        expected_features = None
        expected_seq_len = 24  # 默认序列长度
        is_lightgbm = hasattr(model, 'model') and hasattr(model.model, 'n_features_in_')
        
        if is_lightgbm:
            # LightGBM期望扁平化的特征 (seq_len * n_features)
            expected_flattened = model.model.n_features_in_
            # 计算期望的序列长度
            expected_seq_len = expected_flattened // n_raw_features
            if expected_seq_len * n_raw_features != expected_flattened:
                # 如果不能整除，使用保存的特征数调整
                expected_seq_len = max(1, expected_flattened // max(1, n_raw_features))
            expected_features = n_raw_features
        elif hasattr(model, 'input_shape') and model.input_shape is not None:
            expected_seq_len, expected_features = model.input_shape
        
        # 准备最近的数据
        window_size = expected_seq_len  # 使用模型期望的序列长度
        if len(df_features) < window_size:
            return None, None
        
        # 获取最近的特征
        recent_data = df_features[feature_cols].iloc[-window_size:].values
        
        # 如果特征数不匹配，进行调整
        if expected_features is not None and recent_data.shape[1] != expected_features:
            if recent_data.shape[1] > expected_features:
                # 特征太多，截取前面的
                recent_data = recent_data[:, :expected_features]
            else:
                # 特征太少，补零
                padding = np.zeros((recent_data.shape[0], expected_features - recent_data.shape[1]))
                recent_data = np.hstack([recent_data, padding])
        
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
        import traceback
        st.code(traceback.format_exc())
        return None, None


@st.cache_resource
def get_event_loop():
    """获取或创建持久化的事件循环"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def fetch_realtime_data_sync():
    """获取实时数据（同步版本）- 使用持久化事件循环"""
    try:
        # 使用持久化的事件循环
        loop = get_event_loop()
        
        # 创建一个新的 API 实例并在其中执行
        async def _fetch():
            # 每次创建新的 session
            api = BinancePublicAPI()
            try:
                price_data = await api.get_current_price("BTCUSDT")
                ticker_data = await api.get_ticker_24h("BTCUSDT")
                return price_data, ticker_data
            finally:
                await api.close()
        
        # 如果循环已经在运行，使用 nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except:
            pass
        
        result = loop.run_until_complete(_fetch())
        return result
    except Exception as e:
        import traceback
        st.error(f"获取数据失败: {e}")
        with st.expander("错误详情"):
            st.code(traceback.format_exc())
        return None, None


def fetch_klines_sync(interval: str, days: int):
    """获取 K 线数据（同步版本）- 使用持久化事件循环"""
    try:
        loop = get_event_loop()
        
        async def _fetch():
            api = BinancePublicAPI()
            try:
                result = await api.get_klines("BTCUSDT", interval, days)
                return result
            finally:
                await api.close()
        
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except:
            pass
        
        result = loop.run_until_complete(_fetch())
        return result
    except Exception as e:
        import traceback
        st.error(f"获取K线数据失败: {e}")
        with st.expander("错误详情"):
            st.code(traceback.format_exc())
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
    
    # 初始化模型管理器
    model_manager = ModelManager()
    available_models = model_manager.scan_models()
    
    # 侧边栏设置
    st.sidebar.header("⚙️ 设置")
    
    # 模型选择
    enable_prediction = st.sidebar.checkbox("🤖 启用 AI 预测", value=False)
    selected_model_info: Optional[ModelInfo] = None
    
    if enable_prediction:
        if available_models:
            # 创建模型选项列表
            model_options = {
                f"{m.name} ({m.model_type}, {m.file_size_mb:.1f}MB)": m
                for m in available_models
            }
            selected_key = st.sidebar.selectbox(
                "选择预测模型",
                list(model_options.keys()),
                index=0
            )
            selected_model_info = model_options[selected_key]
            
            # 显示模型详情
            with st.sidebar.expander("📊 模型详情", expanded=False):
                if selected_model_info:
                    st.markdown(f"**模型类型**: {selected_model_info.model_type}")
                    st.markdown(f"**文件大小**: {selected_model_info.file_size_mb:.2f} MB")
                    st.markdown(f"**创建时间**: {selected_model_info.created_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    # 显示模型配置
                    config_items = []
                    if selected_model_info.input_shape:
                        config_items.append(f"  - 输入形状: {selected_model_info.input_shape}")
                    if selected_model_info.hidden_size:
                        config_items.append(f"  - 隐藏层大小: {selected_model_info.hidden_size}")
                    if selected_model_info.num_layers:
                        config_items.append(f"  - 层数: {selected_model_info.num_layers}")
                    if selected_model_info.dropout:
                        config_items.append(f"  - Dropout: {selected_model_info.dropout}")
                    if config_items:
                        st.markdown("**配置参数**:")
                        st.text("\n".join(config_items))
                    
                    # 显示训练指标
                    metrics_items = []
                    if selected_model_info.epochs_trained:
                        metrics_items.append(f"  - 训练轮数: {selected_model_info.epochs_trained}")
                    if selected_model_info.best_val_accuracy:
                        metrics_items.append(f"  - 最佳验证准确率: {selected_model_info.best_val_accuracy:.4f}")
                    if selected_model_info.best_val_loss:
                        metrics_items.append(f"  - 最佳验证损失: {selected_model_info.best_val_loss:.4f}")
                    if metrics_items:
                        st.markdown("**训练指标**:")
                        st.text("\n".join(metrics_items))
        else:
            st.sidebar.warning("⚠️ 未找到已训练的模型，请先运行 train.py")
            selected_model_info = None
    
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
    
    st.sidebar.markdown("---")
    if enable_prediction:
        st.sidebar.info("💡 提示: 预测功能需要先训练模型")
    
    # 加载模型
    model = None
    if enable_prediction and selected_model_info:
        with st.spinner(f"加载 {selected_model_info.name} 模型..."):
            model = model_manager.load_model(selected_model_info.file_path)
            if model is None:
                st.sidebar.warning(f"⚠️ {selected_model_info.name} 模型加载失败")
    
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
        
        # AI 预测区域
        if model is not None:
            st.markdown("---")
            st.header("🎯 AI 趋势预测")
            
            # 获取足够的历史数据用于预测
            with st.spinner("正在获取数据并生成预测..."):
                df_pred = fetch_klines_sync("1h", 7)  # 7天小时数据
                
                if not df_pred.empty:
                    pred_class, pred_proba = make_prediction(model, df_pred)
                    
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
                        st.warning("⚠️ 数据不足，无法进行预测")
                else:
                    st.warning("⚠️ 无法获取历史数据")
        
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
