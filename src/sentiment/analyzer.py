"""
NLP情感分析器
================================
使用CryptoBERT和VADER进行文本情感分析
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import SentimentConfig, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """情感分析结果"""
    text: str
    score: float  # -1 到 +1
    confidence: float
    label: str  # "positive", "negative", "neutral"
    method: str  # "cryptobert" 或 "vader"


class SentimentAnalyzer:
    """
    双轨情感分析器
    
    - CryptoBERT: 用于新闻标题（加密货币专用模型）
    - VADER: 用于社交媒体文本（快速，处理非正式语言）
    """
    
    def __init__(self, use_gpu: bool = True):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        logger.info(f"情感分析器使用设备: {self.device}")
        
        self._cryptobert_model = None
        self._cryptobert_tokenizer = None
        self._vader = None
        
        # 初始化VADER (轻量级，总是加载)
        self._init_vader()
    
    def _init_vader(self):
        """初始化VADER情感分析器"""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("下载VADER词典...")
            nltk.download('vader_lexicon', quiet=True)
        
        self._vader = SentimentIntensityAnalyzer()
        logger.info("VADER情感分析器初始化完成")
    
    def _load_cryptobert(self):
        """懒加载CryptoBERT模型"""
        if self._cryptobert_model is not None:
            return
        
        logger.info(f"加载CryptoBERT模型: {SentimentConfig.CRYPTOBERT_MODEL}")
        
        try:
            self._cryptobert_tokenizer = AutoTokenizer.from_pretrained(
                SentimentConfig.CRYPTOBERT_MODEL
            )
            self._cryptobert_model = AutoModelForSequenceClassification.from_pretrained(
                SentimentConfig.CRYPTOBERT_MODEL
            ).to(self.device)
            self._cryptobert_model.eval()
            
            logger.info("CryptoBERT模型加载完成")
        except Exception as e:
            logger.error(f"CryptoBERT加载失败: {e}")
            logger.warning("将仅使用VADER进行情感分析")
    
    def analyze_with_vader(self, text: str) -> AnalysisResult:
        """
        使用VADER分析文本情感
        
        VADER输出:
        - compound: -1到+1的综合分数
        - pos/neg/neu: 各情感比例
        """
        if self._vader is None:
            return AnalysisResult(
                text=text,
                score=0.0,
                confidence=0.0,
                label="neutral",
                method="vader"
            )
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        
        # 确定标签
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        # 计算置信度 (基于compound的绝对值)
        confidence = abs(compound)
        
        return AnalysisResult(
            text=text,
            score=compound,
            confidence=confidence,
            label=label,
            method="vader"
        )
    
    def analyze_with_cryptobert(self, text: str) -> AnalysisResult:
        """
        使用CryptoBERT分析文本情感
        
        CryptoBERT输出:
        - 3个类别: Bearish, Neutral, Bullish
        """
        self._load_cryptobert()
        
        if self._cryptobert_model is None:
            # 回退到VADER
            return self.analyze_with_vader(text)
        
        # 分词
        if self._cryptobert_tokenizer is None or self._cryptobert_model is None:
            return AnalysisResult(
                text=text,
                score=0.0,
                confidence=0.0,
                label="neutral",
                method="cryptobert"
            )
        
        inputs = self._cryptobert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self._cryptobert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        probs = probs.cpu().numpy()[0]
        
        # CryptoBERT标签映射 (根据模型训练顺序)
        # 通常: 0=Bearish, 1=Neutral, 2=Bullish
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        score_map = {0: -1.0, 1: 0.0, 2: 1.0}
        
        # 获取最高概率的标签
        pred_idx = probs.argmax()
        confidence = float(probs[pred_idx])
        
        # 计算加权分数
        weighted_score = sum(probs[i] * score_map[i] for i in range(len(probs)))
        
        return AnalysisResult(
            text=text,
            score=weighted_score,
            confidence=confidence,
            label=label_map[pred_idx],
            method="cryptobert"
        )
    
    def analyze(
        self, 
        text: str, 
        use_cryptobert: bool = True
    ) -> AnalysisResult:
        """
        分析文本情感
        
        Args:
            text: 待分析文本
            use_cryptobert: 是否使用CryptoBERT (否则用VADER)
        """
        if not text or not text.strip():
            return AnalysisResult(
                text=text,
                score=0.0,
                confidence=0.0,
                label="neutral",
                method="none"
            )
        
        if use_cryptobert:
            return self.analyze_with_cryptobert(text)
        else:
            return self.analyze_with_vader(text)
    
    def analyze_batch(
        self, 
        texts: List[str],
        use_cryptobert: bool = True
    ) -> List[AnalysisResult]:
        """
        批量分析文本情感
        
        对于CryptoBERT，批量处理更高效
        """
        if not texts:
            return []
        
        if not use_cryptobert or self._cryptobert_model is None:
            return [self.analyze_with_vader(t) for t in texts]
        
        self._load_cryptobert()
        
        if self._cryptobert_model is None:
            return [self.analyze_with_vader(t) for t in texts]
        
        results = []
        batch_size = 16  # GTX 1650优化
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 批量分词
            if self._cryptobert_tokenizer is None:
                results.extend([self.analyze_with_vader(t) for t in batch_texts])
                continue
                
            inputs = self._cryptobert_tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # 批量推理
            with torch.no_grad():
                outputs = self._cryptobert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            probs = probs.cpu().numpy()
            
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            score_map = {0: -1.0, 1: 0.0, 2: 1.0}
            
            for j, text in enumerate(batch_texts):
                p = probs[j]
                pred_idx = p.argmax()
                confidence = float(p[pred_idx])
                weighted_score = sum(p[k] * score_map[k] for k in range(len(p)))
                
                results.append(AnalysisResult(
                    text=text,
                    score=weighted_score,
                    confidence=confidence,
                    label=label_map[pred_idx],
                    method="cryptobert"
                ))
        
        return results
    
    def get_aggregate_sentiment(
        self, 
        results: List[AnalysisResult]
    ) -> Tuple[float, float]:
        """
        聚合多个分析结果
        
        Returns:
            (平均分数, 平均置信度)
        """
        if not results:
            return 0.0, 0.0
        
        total_score = sum(r.score * r.confidence for r in results)
        total_confidence = sum(r.confidence for r in results)
        
        if total_confidence == 0:
            return 0.0, 0.0
        
        avg_score = total_score / total_confidence
        avg_confidence = total_confidence / len(results)
        
        return avg_score, avg_confidence


# 全局分析器实例
sentiment_analyzer = SentimentAnalyzer(use_gpu=True)
