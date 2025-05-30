#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版歌词预测器
解决预测结果重复和质量问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import numpy as np
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm


class OptimizedLyricsDataset(Dataset):
    """优化的歌词数据集"""
    def __init__(self, sequences, vocab_to_idx, seq_length=12):
        self.vocab_to_idx = vocab_to_idx
        self.seq_length = seq_length
        
        # 过滤和标准化序列
        self.sequences = []
        for seq in sequences:
            if len(seq) >= 2:  # 至少需要2个词（输入和目标）
                # 如果序列太长，截断；如果太短，保持原样
                if len(seq) > seq_length + 1:
                    normalized_seq = seq[:seq_length + 1]
                else:
                    normalized_seq = seq
                self.sequences.append(normalized_seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        
        # 转换为索引
        input_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in input_seq]
        target_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in target_seq]
        
        # 填充到固定长度
        while len(input_indices) < self.seq_length:
            input_indices.append(self.vocab_to_idx['<PAD>'])
        while len(target_indices) < self.seq_length:
            target_indices.append(self.vocab_to_idx['<PAD>'])
        
        # 截断到固定长度
        input_indices = input_indices[:self.seq_length]
        target_indices = target_indices[:self.seq_length]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)


class OptimizedLyricsLSTM(nn.Module):
    """优化的LSTM模型"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(OptimizedLyricsLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 应用注意力机制
        lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        
        # 结合LSTM输出和注意力输出
        combined = lstm_out + attn_out
        combined = self.dropout(combined)
        output = self.fc(combined)
        
        return output, hidden


class OptimizedLyricsPredictor:
    """优化版歌词预测器"""
    
    def __init__(self, model_path='optimized_lyrics_model.pth', vocab_path='optimized_vocab.pkl'):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.seq_length = 12
        
        # 预定义的高质量歌词模板
        self.quality_templates = [
            "{emotion}的{time}里，{action}{object}",
            "{place}的{object}{verb}，{feeling}在心中",
            "{adjective}的{noun}如{metaphor}，{description}",
            "当{condition}的时候，{result}",
            "{subject}{verb}{object}，{emotion}{feeling}",
        ]
        
        # 词汇分类字典
        self.word_categories = {
            'emotion': ['温暖', '甜美', '浪漫', '深情', '纯真', '美好'],
            'time': ['春天', '夏日', '秋天', '冬季', '清晨', '黄昏', '夜晚'],
            'place': ['山间', '海边', '林中', '花园', '桥边', '湖畔', '草原'],
            'object': ['花朵', '月亮', '星空', '河流', '微风', '阳光', '彩虹'],
            'verb': ['盛开', '照耀', '流淌', '飞翔', '歌唱', '舞蹈', '闪烁'],
            'feeling': ['思念', '温暖', '幸福', '快乐', '感动', '宁静', '喜悦'],
            'adjective': ['美丽', '清新', '动人', '迷人', '绚丽', '可爱'],
            'noun': ['梦想', '希望', '友谊', '爱情', '青春', '回忆', '未来'],
            'metaphor': ['流水', '春风', '朝霞', '星辰', '花香', '音符'],
        }
    
    def get_enhanced_lyrics_data(self):
        """获取增强的歌词数据"""
        lyrics_data = [
            # 经典诗词风格
            "春江花月夜，明月几时有",
            "山重水复疑无路，柳暗花明又一村", 
            "海内存知己，天涯若比邻",
            "落红不是无情物，化作春泥更护花",
            "采菊东篱下，悠然见南山",
            "会当凌绝顶，一览众山小",
            "床前明月光，疑是地上霜",
            "春眠不觉晓，处处闻啼鸟",
            "白日依山尽，黄河入海流",
            "独在异乡为异客，每逢佳节倍思亲",
            
            # 现代流行风格
            "你是我心中最美的风景",
            "时间会带走一切烦恼",
            "每一个明天都充满希望",
            "梦想的翅膀带我飞翔",
            "微笑是最美的语言",
            "相遇是美丽的缘分",
            "岁月如歌声声入耳",
            "青春不散场梦想不打烊",
            "星光点点照亮夜空",
            "风雨过后见彩虹",
            "爱如春风暖人心",
            "情如细雨润心田",
            "勇敢面对每一天",
            "心中有爱不孤单",
            "眼中有光不迷茫",
            "友谊如花永远开",
            "真情如水永远流",
            "时光飞逝如流水",
            "岁月如歌永流传",
            
            # 情感抒发
            "思君不见君想君君不知",
            "相思如潮水思念如流云",
            "千里共婵娟但愿人长久",
            "春花秋月何时了",
            "最美不过初相见",
            "情深深雨蒙蒙",
            "红豆生南国春来发几枝",
            "愿得一心人白头不相离",
            "离别总是在九月",
            "回忆是思念的愁",
            "最苦不过离别时",
            "多少楼台烟雨中",
            "山无陵江水为竭",
            "冬雷震震夏雨雪",
            
            # 励志正能量
            "相信自己一定能行",
            "困难面前不低头",
            "每一次跌倒都是成长",
            "勇敢追求心中的梦",
            "今天的努力是明天的收获",
            "机会总是留给有准备的人",
            "只要心中有阳光",
            "用微笑面对生活",
            "奋斗的青春最美丽",
            "坚持到底就是胜利",
            "挫折面前不气馁",
            "每一次失败都是经验",
            "永远不要说放弃",
            "今天的汗水是明天的彩虹",
            "成功总是属于坚持的人",
            "路虽远行则将至",
            "事虽难做则必成",
            "生活就会有希望",
            "用真诚对待朋友",
            "拼搏的人生最精彩",
            
            # 自然风光
            "春暖花开万物苏",
            "秋高气爽丹桂香",
            "小溪潺潺流向海",
            "高山巍峨入云端",
            "森林深处鸟语花香",
            "海浪拍岸声声美",
            "朝霞满天红似火",
            "星辰大海任遨游",
            "花开花落有时节",
            "山清水秀好风光",
            "夏日炎炎绿荫浓",
            "冬雪纷飞梅花开",
            "大河滔滔奔向前",
            "深谷幽静藏仙境",
            "草原辽阔马儿奔跑",
            "海鸥翱翔天地间",
            "夕阳西下金满山",
            "云卷云舒自悠然",
            "云来云去无定所",
            "鸟语花香醉人心",
            
            # 民族风格
            "茉莉花开满园香",
            "采花姑娘心欢畅",
            "山歌好比春江水",
            "不怕滩险弯又多",
            "阿里山的姑娘美如水",
            "阿里山的少年壮如山",
            "草原上升起不落的太阳",
            "照耀着我们美丽的家乡",
            "北风那个吹雪花那个飘",
            "江南烟雨蒙蒙",
            "小桥流水人家",
            "黄土高原唱山歌",
            "信天游声震山河",
            "竹林深处有人家",
            "翠绿竹叶映朝霞",
            "渔舟唱晚满江红",
            "夕阳西下水波清",
            "梅花朵朵开满山",
            "雪花片片舞满天",
            "桃花源里好风光",
            "世外桃源在心中",
            "春风又绿江南岸",
            "明月何时照我还",
            "塞外风光无限好",
            "草原儿女情意长",
            
            # 生活感悟
            "平凡的生活也有诗意",
            "简单的日子也有美丽",
            "家是心灵的港湾",
            "爱是生命的源泉",
            "健康是最大的财富",
            "快乐是最好的礼物",
            "珍惜眼前的幸福",
            "感恩身边的温暖",
            "生活如茶需要慢慢品",
            "人生如书需要细细读",
            "知足常乐心自在",
            "助人为乐情自真",
            "做人如水能载舟",
            "做事如山能担当",
            "真诚待人人待真",
            "善良处世世处善",
            "宽容是智慧的体现",
            "理解是友谊的基础",
            "今天很残酷明天更残酷",
            "后天很美好",
        ]
        
        # 生成更多变体
        extended_data = []
        for lyric in lyrics_data:
            extended_data.append(lyric)
            # 为每个歌词生成1-2个变体
            variations = self.generate_lyric_variations(lyric)
            extended_data.extend(variations[:2])
        
        return extended_data
    
    def generate_lyric_variations(self, original_lyric):
        """生成歌词变体"""
        variations = []
        words = list(jieba.cut(original_lyric))
        
        # 同义词替换
        synonyms = {
            '美丽': ['美好', '漂亮', '动人', '迷人'],
            '温暖': ['温馨', '暖和', '温柔', '亲切'],
            '快乐': ['开心', '愉快', '高兴', '欢乐'],
            '梦想': ['理想', '心愿', '愿望', '憧憬'],
            '希望': ['期望', '盼望', '渴望', '企盼'],
            '爱情': ['真爱', '深情', '恋情', '爱意'],
        }
        
        # 生成替换变体
        for i, word in enumerate(words):
            if word in synonyms:
                for synonym in synonyms[word][:2]:  # 最多2个同义词
                    new_words = words.copy()
                    new_words[i] = synonym
                    variation = ''.join(new_words)
                    if variation != original_lyric:
                        variations.append(variation)
        
        return variations
    
    def preprocess_text(self, text):
        """文本预处理"""
        if not text:
            return []
        
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）\s]', '', text)
        
        # 使用jieba分词
        words = list(jieba.cut(text.strip()))
        
        # 过滤和清理
        cleaned_words = []
        for word in words:
            word = word.strip()
            if word and len(word) > 0:
                cleaned_words.append(word)
        
        return cleaned_words
    
    def build_optimized_vocab(self):
        """构建优化的词汇表"""
        print("正在构建优化词汇表...")
        
        # 获取歌词数据
        lyrics_data = self.get_enhanced_lyrics_data()
        print(f"收集到 {len(lyrics_data)} 条歌词数据")
        
        # 预处理所有歌词
        all_words = []
        processed_lyrics = []
        
        for lyric in lyrics_data:
            words = self.preprocess_text(lyric)
            if len(words) >= 3:
                all_words.extend(words)
                processed_lyrics.append(words)
        
        # 统计词频
        word_counts = Counter(all_words)
        
        # 构建词汇表
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # 添加高频词（出现次数>=2）
        frequent_words = [word for word, count in word_counts.most_common() if count >= 2]
        vocab.extend(frequent_words)
        
        # 添加重要的单字词
        important_chars = ['的', '了', '在', '是', '有', '和', '与', '或', '但', '却', '也', '都', '很', '更', '最']
        for char in important_chars:
            if char not in vocab:
                vocab.append(char)
        
        self.vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_vocab = {idx: word for word, idx in self.vocab_to_idx.items()}
        
        # 保存词汇表
        with open(self.vocab_path, 'wb') as f:
            pickle.dump((self.vocab_to_idx, self.idx_to_vocab), f)
        
        print(f"词汇表构建完成，共包含 {len(vocab)} 个词")
        return processed_lyrics
    
    def prepare_training_data(self, lyrics_list):
        """准备训练数据"""
        sequences = []
        
        for lyric_words in lyrics_list:
            if len(lyric_words) < 2:  # 降低最小长度要求
                continue
            
            # 添加特殊标记
            words = ['<START>'] + lyric_words + ['<END>']
            
            # 创建多种长度的滑动窗口序列
            max_len = min(len(words) - 1, self.seq_length)
            for seq_len in range(3, max_len + 1):  # 从3到最大长度
                for i in range(len(words) - seq_len):
                    sequence = words[i:i + seq_len + 1]
                    sequences.append(sequence)
        
        return sequences
    
    def train_optimized_model(self, epochs=80, batch_size=16, learning_rate=0.001):
        """训练优化模型"""
        print("开始训练优化歌词预测模型...")
        
        # 构建词汇表和准备数据
        processed_lyrics = self.build_optimized_vocab()
        sequences = self.prepare_training_data(processed_lyrics)
        
        if not sequences:
            print("没有足够的训练数据")
            return
        
        print(f"词汇表大小: {len(self.vocab_to_idx)}")
        print(f"训练序列数量: {len(sequences)}")
        
        # 创建数据集和数据加载器
        dataset = OptimizedLyricsDataset(sequences, self.vocab_to_idx, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        vocab_size = len(self.vocab_to_idx)
        self.model = OptimizedLyricsLSTM(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3
        )
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_to_idx['<PAD>'])
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
        
        # 训练循环
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_input, batch_target in progress_bar:
                optimizer.zero_grad()
                
                output, _ = self.model(batch_input)
                loss = criterion(output.reshape(-1, vocab_size), batch_target.reshape(-1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_path)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}')
        
        print(f"训练完成！最佳模型已保存到 {self.model_path}")
    
    def load_model(self):
        """加载模型"""
        try:
            with open(self.vocab_path, 'rb') as f:
                self.vocab_to_idx, self.idx_to_vocab = pickle.load(f)
            
            vocab_size = len(self.vocab_to_idx)
            self.model = OptimizedLyricsLSTM(vocab_size=vocab_size)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            print("优化模型加载成功！")
            return True
        except FileNotFoundError:
            print("模型文件不存在，请先训练模型")
            return False
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def predict_with_quality_control(self, input_line, num_predictions=3):
        """带质量控制的预测"""
        if self.model is None:
            return ["模型未加载"]
        
        try:
            predictions = []
            
            # 方法1: 基于模型的预测
            model_predictions = self._model_based_prediction(input_line, num_predictions)
            predictions.extend(model_predictions)
            
            # 方法2: 基于模板的预测
            if len(predictions) < num_predictions:
                template_predictions = self._template_based_prediction(input_line, num_predictions - len(predictions))
                predictions.extend(template_predictions)
            
            # 方法3: 基于规则的预测
            if len(predictions) < num_predictions:
                rule_predictions = self._rule_based_prediction(input_line, num_predictions - len(predictions))
                predictions.extend(rule_predictions)
            
            # 去重和质量过滤
            unique_predictions = []
            for pred in predictions:
                if pred not in unique_predictions and len(pred) > 2:
                    unique_predictions.append(pred)
            
            return unique_predictions[:num_predictions] if unique_predictions else ["暂无合适的预测结果"]
            
        except Exception as e:
            return [f"预测出错: {str(e)}"]
    
    def _model_based_prediction(self, input_line, num_predictions):
        """基于模型的预测"""
        words = self.preprocess_text(input_line)
        if not words:
            return []
        
        predictions = []
        
        for i in range(num_predictions):
            # 使用不同的温度和随机种子
            temperature = 0.8 + (i * 0.1)
            torch.manual_seed(42 + i * 7)
            
            input_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in words]
            
            if len(input_indices) > self.seq_length:
                input_indices = input_indices[-self.seq_length:]
            
            while len(input_indices) < self.seq_length:
                input_indices.insert(0, self.vocab_to_idx['<PAD>'])
            
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor([input_indices])
                hidden = None
                predicted_words = []
                
                for step in range(15):  # 最多生成15个词
                    output, hidden = self.model(input_tensor, hidden)
                    logits = output[0, -1, :] / temperature
                    
                    # Top-k采样
                    k = max(3, 8 - i)
                    top_k_logits, top_k_indices = torch.topk(logits, k)
                    top_k_probs = torch.softmax(top_k_logits, dim=0)
                    
                    selected_idx = torch.multinomial(top_k_probs, 1).item()
                    next_word_idx = top_k_indices[selected_idx].item()
                    next_word = self.idx_to_vocab[next_word_idx]
                    
                    if next_word in ['<END>', '<PAD>']:
                        break
                    
                    if next_word != '<UNK>' and (not predicted_words or next_word != predicted_words[-1]):
                        predicted_words.append(next_word)
                    
                    input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_word_idx]])], dim=1)
                    
                    # 早停条件
                    if len(predicted_words) >= 6 and any(punct in ''.join(predicted_words) for punct in ['，', '。', '！', '？']):
                        break
                
                result = ''.join(predicted_words)
                if result and len(result) > 2:
                    predictions.append(result)
        
        return predictions
    
    def _template_based_prediction(self, input_line, num_predictions):
        """基于模板的预测"""
        predictions = []
        
        for i in range(num_predictions):
            template = random.choice(self.quality_templates)
            
            # 随机填充模板
            filled_template = template
            for category, words in self.word_categories.items():
                placeholder = f"{{{category}}}"
                if placeholder in filled_template:
                    replacement = random.choice(words)
                    filled_template = filled_template.replace(placeholder, replacement, 1)
            
            # 清理模板中剩余的占位符
            filled_template = re.sub(r'\{[^}]+\}', '', filled_template)
            
            if filled_template and len(filled_template) > 2:
                predictions.append(filled_template)
        
        return predictions
    
    def _rule_based_prediction(self, input_line, num_predictions):
        """基于规则的预测"""
        predictions = []
        
        # 简单的规则生成
        rule_patterns = [
            "如花般美丽绽放",
            "像星星一样闪亮",
            "温暖如春风拂面",
            "清新如晨露甘甜",
            "深情如海洋无边",
            "纯真如白云飘逸",
        ]
        
        for i in range(min(num_predictions, len(rule_patterns))):
            predictions.append(rule_patterns[i])
        
        return predictions
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🎵 优化版歌词预测系统")
        print("=" * 50)
        print("输入一句歌词，AI将为您创作下一句")
        print("输入 'quit' 退出程序")
        print("=" * 50)
        
        while True:
            user_input = input("\n🎤 请输入一句歌词: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 感谢使用！")
                break
            
            if not user_input:
                print("请输入有效内容")
                continue
            
            print(f"\n📝 输入: {user_input}")
            predictions = self.predict_with_quality_control(user_input)
            
            print("🎼 AI创作的下一句:")
            for i, pred in enumerate(predictions, 1):
                print(f"   {i}. {pred}")
            print("-" * 50)


def main():
    """主函数"""
    predictor = OptimizedLyricsPredictor()
    
    print("🎵 优化版歌词预测系统")
    print("=" * 40)
    print("1. 训练新的优化模型")
    print("2. 加载模型并开始预测")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == '1':
        print("开始训练优化模型...")
        predictor.train_optimized_model(epochs=60)
        
        if predictor.load_model():
            predictor.interactive_mode()
    
    elif choice == '2':
        if predictor.load_model():
            predictor.interactive_mode()
        else:
            print("模型加载失败！请先训练模型。")
    
    else:
        print("无效选择！")


if __name__ == "__main__":
    main() 