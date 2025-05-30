import torch
import torch.nn as nn
import torch.optim as optim
import jieba
import numpy as np
import json
import os
import pickle
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm


class LyricsDataset(Dataset):
    """歌词数据集类"""
    def __init__(self, sequences, vocab_to_idx, seq_length=10):
        self.sequences = sequences
        self.vocab_to_idx = vocab_to_idx
        self.seq_length = seq_length
        
        # 过滤和调整序列长度
        self.filtered_sequences = []
        for seq in sequences:
            if len(seq) >= seq_length + 1:  # 需要至少seq_length+1个词来构建输入和目标
                # 截取到固定长度
                adjusted_seq = seq[:seq_length + 1]
                self.filtered_sequences.append(adjusted_seq)
        
    def __len__(self):
        return len(self.filtered_sequences)
    
    def __getitem__(self, idx):
        sequence = self.filtered_sequences[idx]
        input_seq = sequence[:-1]  # 前seq_length个词作为输入
        target_seq = sequence[1:]  # 后seq_length个词作为目标
        
        # 确保长度一致
        assert len(input_seq) == self.seq_length
        assert len(target_seq) == self.seq_length
        
        # 转换为索引
        input_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in input_seq]
        target_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in target_seq]
        
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)


class LyricsLSTM(nn.Module):
    """LSTM歌词预测模型"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(LyricsLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden


class LyricsPredictor:
    """歌词预测器主类"""
    def __init__(self, model_path='lyrics_model.pth', vocab_path='vocab.pkl'):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.seq_length = 10
        
    def preprocess_text(self, text):
        """文本预处理"""
        # 去除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）\[\]【】\s]', '', text)
        # 使用jieba分词
        words = list(jieba.cut(text.strip()))
        # 过滤空白词
        words = [word.strip() for word in words if word.strip()]
        return words
    
    def build_vocab_from_sample_lyrics(self):
        """从示例歌词构建词汇表"""
        # 一些示例歌词数据（避免版权问题，使用简化的示例）
        sample_lyrics = [
            "月亮代表我的心 你问我爱你有多深",
            "春天在哪里呀 春天在哪里 春天在那青翠的山林里",
            "让我们荡起双桨 小船儿推开波浪",
            "我爱你中国 我爱你春天蓬勃的秧苗",
            "歌声飞扬 梦想起航 青春无悔",
            "阳光灿烂的日子 微风轻抚面庞",
            "友谊之光照亮前方 温暖的话语在心中",
            "时光荏苒岁月如歌 美好回忆永不褪色",
            "勇敢追寻心中梦想 不怕风雨阻挡前进",
            "花开花落又一年 相聚离别在人间"
        ]
        
        all_words = []
        for lyric in sample_lyrics:
            words = self.preprocess_text(lyric)
            all_words.extend(words)
        
        # 统计词频并构建词汇表
        word_counts = Counter(all_words)
        
        # 添加特殊token
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab.extend([word for word, count in word_counts.most_common() if count >= 1])
        
        self.vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_vocab = {idx: word for word, idx in self.vocab_to_idx.items()}
        
        # 保存词汇表
        with open(self.vocab_path, 'wb') as f:
            pickle.dump((self.vocab_to_idx, self.idx_to_vocab), f)
        
        return sample_lyrics
    
    def prepare_training_data(self, lyrics_list):
        """准备训练数据"""
        sequences = []
        
        for lyric in lyrics_list:
            words = self.preprocess_text(lyric)
            if len(words) < 3:  # 跳过太短的歌词
                continue
                
            # 添加开始和结束token
            words = ['<START>'] + words + ['<END>']
            
            # 创建序列
            for i in range(len(words) - self.seq_length):
                sequence = words[i:i + self.seq_length + 1]
                sequences.append(sequence)
        
        return sequences
    
    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001):
        """训练模型"""
        print("正在构建词汇表和准备数据...")
        
        # 构建词汇表和获取示例数据
        sample_lyrics = self.build_vocab_from_sample_lyrics()
        
        # 准备训练数据
        sequences = self.prepare_training_data(sample_lyrics)
        
        if not sequences:
            print("没有找到足够的训练数据")
            return
        
        print(f"词汇表大小: {len(self.vocab_to_idx)}")
        print(f"训练序列数量: {len(sequences)}")
        
        # 创建数据加载器
        dataset = LyricsDataset(sequences, self.vocab_to_idx, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型
        vocab_size = len(self.vocab_to_idx)
        self.model = LyricsLSTM(vocab_size)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_to_idx['<PAD>'])
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_input, batch_target in dataloader:
                optimizer.zero_grad()
                
                output, _ = self.model(batch_input)
                loss = criterion(output.reshape(-1, vocab_size), batch_target.reshape(-1))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        print(f"模型已保存到 {self.model_path}")
    
    def load_model(self):
        """加载已训练的模型"""
        try:
            # 加载词汇表
            with open(self.vocab_path, 'rb') as f:
                self.vocab_to_idx, self.idx_to_vocab = pickle.load(f)
            
            # 初始化并加载模型
            vocab_size = len(self.vocab_to_idx)
            self.model = LyricsLSTM(vocab_size)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            print("模型加载成功！")
            return True
        except FileNotFoundError:
            print("模型文件不存在，请先训练模型")
            return False
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def predict_next_line(self, input_line, max_length=20, temperature=0.8):
        """预测下一句歌词"""
        if self.model is None:
            print("请先加载模型")
            return ""
        
        # 预处理输入
        words = self.preprocess_text(input_line)
        if not words:
            return "输入无效，请提供有意义的歌词"
        
        # 转换为索引
        input_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in words]
        
        # 如果输入太长，只取最后几个词
        if len(input_indices) > self.seq_length:
            input_indices = input_indices[-self.seq_length:]
        
        # 补充到固定长度
        while len(input_indices) < self.seq_length:
            input_indices.insert(0, self.vocab_to_idx['<PAD>'])
        
        # 生成预测
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor([input_indices])
            hidden = None
            predicted_words = []
            
            for _ in range(max_length):
                output, hidden = self.model(input_tensor, hidden)
                
                # 应用温度采样
                logits = output[0, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=0)
                
                # 随机采样
                next_word_idx = torch.multinomial(probabilities, 1).item()
                next_word = self.idx_to_vocab[next_word_idx]
                
                # 如果遇到结束符或填充符，停止生成
                if next_word in ['<END>', '<PAD>']:
                    break
                
                if next_word != '<UNK>':
                    predicted_words.append(next_word)
                
                # 更新输入序列
                input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_word_idx]])], dim=1)
            
            return ''.join(predicted_words) if predicted_words else "无法生成预测"


def main():
    """主函数"""
    predictor = LyricsPredictor()
    
    print("=== 歌词预测系统 ===")
    print("1. 训练新模型")
    print("2. 加载已有模型并预测")
    
    choice = input("请选择操作 (1/2): ").strip()
    
    if choice == '1':
        print("开始训练模型...")
        predictor.train_model(epochs=50)
        
        # 训练完成后可以进行预测
        print("\n训练完成！现在可以进行预测了。")
        while True:
            input_line = input("\n请输入一句歌词 (输入 'quit' 退出): ").strip()
            if input_line.lower() == 'quit':
                break
            
            prediction = predictor.predict_next_line(input_line)
            print(f"预测的下一句: {prediction}")
    
    elif choice == '2':
        if predictor.load_model():
            print("模型加载成功！可以开始预测了。")
            while True:
                input_line = input("\n请输入一句歌词 (输入 'quit' 退出): ").strip()
                if input_line.lower() == 'quit':
                    break
                
                prediction = predictor.predict_next_line(input_line)
                print(f"预测的下一句: {prediction}")
        else:
            print("模型加载失败！")
    
    else:
        print("无效选择！")


if __name__ == "__main__":
    main() 