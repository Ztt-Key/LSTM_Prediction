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
import re
from tqdm import tqdm
from lyrics_collector import LyricsCollector


class EnhancedLyricsPredictor:
    """增强版歌词预测器"""
    
    def __init__(self, model_path='enhanced_lyrics_model.pth', vocab_path='enhanced_vocab.pkl'):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.model = None
        self.vocab_to_idx = {}
        self.idx_to_vocab = {}
        self.seq_length = 15  # 增加序列长度
        self.collector = LyricsCollector()
        
    def preprocess_text(self, text):
        """文本预处理"""
        # 去除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）\[\]【】\s]', '', text)
        # 使用jieba分词
        words = list(jieba.cut(text.strip()))
        # 过滤空白词
        words = [word.strip() for word in words if word.strip()]
        return words
    
    def build_enhanced_vocab(self):
        """构建增强词汇表"""
        print("正在构建增强词汇表...")
        
        # 首先尝试使用增强版数据收集器
        try:
            from enhanced_data_collector import EnhancedDataCollector
            enhanced_collector = EnhancedDataCollector()
            enhanced_data = enhanced_collector.collect_enhanced_data()
            print(f"使用增强版数据集，共 {len(enhanced_data)} 条数据")
        except:
            print("增强版数据收集器不可用，使用原始数据收集器...")
            enhanced_data = self.collector.expand_lyrics_dataset()
        
        # 预处理所有歌词
        all_words = []
        processed_lyrics = []
        
        for lyric in enhanced_data:
            words = self.preprocess_text(lyric)
            if len(words) >= 3:  # 至少3个词才有意义
                all_words.extend(words)
                processed_lyrics.append(words)
        
        # 统计词频并构建词汇表
        word_counts = Counter(all_words)
        
        # 添加特殊token
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        # 添加高频词（出现次数>=2的词）
        frequent_words = [word for word, count in word_counts.most_common() if count >= 2]
        vocab.extend(frequent_words)
        
        self.vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_vocab = {idx: word for word, idx in self.vocab_to_idx.items()}
        
        # 保存词汇表
        with open(self.vocab_path, 'wb') as f:
            pickle.dump((self.vocab_to_idx, self.idx_to_vocab), f)
        
        print(f"词汇表构建完成，共包含 {len(vocab)} 个词")
        return processed_lyrics
    
    def prepare_enhanced_training_data(self, lyrics_list):
        """准备增强训练数据"""
        sequences = []
        
        for lyric in lyrics_list:
            # 如果lyric是列表，先转换为字符串
            if isinstance(lyric, list):
                lyric_text = ''.join(lyric)
            else:
                lyric_text = lyric
                
            words = self.preprocess_text(lyric_text)
            if len(words) < 3:  # 跳过太短的歌词
                continue
                
            # 添加开始和结束token
            words = ['<START>'] + words + ['<END>']
            
            # 创建多种长度的序列
            for seq_len in [self.seq_length - 2, self.seq_length, self.seq_length + 2]:
                for i in range(len(words) - seq_len):
                    sequence = words[i:i + seq_len + 1]
                    sequences.append(sequence)
        
        return sequences
    
    def train_enhanced_model(self, epochs=80, batch_size=16, learning_rate=0.001):
        """训练增强模型"""
        print("开始训练增强歌词预测模型...")
        
        # 构建词汇表和获取数据
        all_lyrics = self.build_enhanced_vocab()
        
        # 准备训练数据
        sequences = self.prepare_enhanced_training_data(all_lyrics)
        
        if not sequences:
            print("没有找到足够的训练数据")
            return
        
        print(f"词汇表大小: {len(self.vocab_to_idx)}")
        print(f"训练序列数量: {len(sequences)}")
        
        # 创建数据集和数据加载器
        from lyrics_predictor import LyricsDataset, LyricsLSTM
        
        dataset = LyricsDataset(sequences, self.vocab_to_idx, self.seq_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 初始化模型（增加复杂度）
        vocab_size = len(self.vocab_to_idx)
        self.model = LyricsLSTM(
            vocab_size=vocab_size, 
            embedding_dim=256,  # 增加嵌入维度
            hidden_dim=512,     # 增加隐藏层维度
            num_layers=3,       # 增加层数
            dropout=0.4
        )
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_to_idx['<PAD>'])
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        
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
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), self.model_path)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}')
        
        print(f"训练完成！最佳模型已保存到 {self.model_path}")
    
    def load_model(self):
        """加载已训练的模型"""
        try:
            # 加载词汇表
            with open(self.vocab_path, 'rb') as f:
                self.vocab_to_idx, self.idx_to_vocab = pickle.load(f)
            
            # 初始化并加载模型
            from lyrics_predictor import LyricsLSTM
            vocab_size = len(self.vocab_to_idx)
            self.model = LyricsLSTM(
                vocab_size=vocab_size, 
                embedding_dim=256,
                hidden_dim=512,
                num_layers=3,
                dropout=0.4
            )
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            print("增强模型加载成功！")
            return True
        except FileNotFoundError:
            print("模型文件不存在，请先训练模型")
            return False
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    
    def predict_next_line_enhanced(self, input_line, max_length=25, temperature=0.7, num_predictions=3):
        """增强预测功能，生成多个候选结果"""
        if self.model is None:
            print("请先加载模型")
            return ["模型未加载"]
        
        try:
            # 预处理输入
            words = self.preprocess_text(input_line)
            if not words:
                return ["输入无效，请提供有意义的歌词"]
            
            # 转换为索引
            input_indices = [self.vocab_to_idx.get(word, self.vocab_to_idx['<UNK>']) for word in words]
            
            # 如果输入太长，只取最后几个词
            if len(input_indices) > self.seq_length:
                input_indices = input_indices[-self.seq_length:]
            
            # 补充到固定长度
            while len(input_indices) < self.seq_length:
                input_indices.insert(0, self.vocab_to_idx['<PAD>'])
            
            predictions = []
            
            # 生成多个不同的预测结果
            for i in range(num_predictions):
                # 为每次预测使用不同的温度和随机种子
                current_temperature = temperature + (i * 0.1)  # 递增温度增加多样性
                if current_temperature > 1.5:
                    current_temperature = 0.5 + (i * 0.05)  # 避免过高温度
                
                # 使用不同的随机种子
                import torch
                torch.manual_seed(42 + i * 10)
                
                self.model.eval()
                with torch.no_grad():
                    input_tensor = torch.tensor([input_indices])
                    hidden = None
                    predicted_words = []
                    
                    # 增加随机起始策略
                    if i > 0:  # 第一个预测保持原始输入，后续预测添加随机扰动
                        # 随机替换最后一个词（如果不是关键词）
                        last_word_idx = input_indices[-1]
                        if last_word_idx not in [self.vocab_to_idx.get('<START>', 0), 
                                               self.vocab_to_idx.get('<END>', 1),
                                               self.vocab_to_idx.get('<PAD>', 2)]:
                            # 从词汇表中随机选择一个相近的词
                            vocab_size = len(self.vocab_to_idx)
                            random_shift = torch.randint(-5, 6, (1,)).item()
                            new_idx = (last_word_idx + random_shift) % vocab_size
                            input_indices_copy = input_indices.copy()
                            input_indices_copy[-1] = new_idx
                            input_tensor = torch.tensor([input_indices_copy])
                    
                    for step in range(max_length):
                        output, hidden = self.model(input_tensor, hidden)
                        
                        # 应用不同的采样策略
                        logits = output[0, -1, :] / current_temperature
                        
                        # 加入top-k采样增加多样性
                        k = max(5, 10 - i)  # 不同预测使用不同的k值
                        top_k_logits, top_k_indices = torch.topk(logits, k)
                        top_k_probs = torch.softmax(top_k_logits, dim=0)
                        
                        # 从top-k中随机采样
                        selected_idx = torch.multinomial(top_k_probs, 1).item()
                        next_word_idx = top_k_indices[selected_idx].item()
                        
                        next_word = self.idx_to_vocab[next_word_idx]
                        
                        # 如果遇到结束符或填充符，停止生成
                        if next_word in ['<END>', '<PAD>']:
                            break
                        
                        # 跳过未知词和重复词
                        if next_word != '<UNK>' and (not predicted_words or next_word != predicted_words[-1]):
                            predicted_words.append(next_word)
                        
                        # 如果连续生成相同的词，引入随机性
                        if len(predicted_words) >= 2 and predicted_words[-1] == predicted_words[-2]:
                            # 随机跳过这个词或选择备选词
                            if torch.rand(1).item() > 0.5:
                                predicted_words.pop()  # 移除重复词
                                # 选择第二个最可能的词
                                if k > 1:
                                    alt_idx = torch.multinomial(top_k_probs, 1).item()
                                    alt_word_idx = top_k_indices[alt_idx].item()
                                    alt_word = self.idx_to_vocab[alt_word_idx]
                                    if alt_word not in ['<UNK>', '<END>', '<PAD>']:
                                        predicted_words.append(alt_word)
                        
                        # 更新输入序列
                        input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_word_idx]])], dim=1)
                        
                        # 提前结束条件：如果生成了合理长度的句子
                        if step >= 8 and len(predicted_words) >= 5:
                            # 检查是否形成了完整的意思
                            result_text = ''.join(predicted_words)
                            if any(punct in result_text for punct in ['，', '。', '！', '？']):
                                break
                    
                    result = ''.join(predicted_words) if predicted_words else f"创意预测{i+1}"
                    
                    # 避免完全相同的结果
                    if result not in predictions:
                        predictions.append(result)
                    else:
                        # 如果结果重复，添加变化
                        modified_result = self._add_variation(result, i)
                        predictions.append(modified_result)
            
            return predictions if predictions else ["无法生成预测，请尝试其他输入"]
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            return [f"预测失败: {str(e)}"]
    
    def _add_variation(self, text, variation_index):
        """为重复的预测结果添加变化"""
        try:
            words = list(jieba.cut(text))
            if len(words) < 2:
                return f"{text}（变化{variation_index + 1}）"
            
            # 简单的变化策略：替换或添加词汇
            variation_words = ['美好', '温暖', '清新', '动人', '甜美', '浪漫', '梦幻', '纯真']
            transition_words = ['如', '像', '似', '若', '仿佛', '恰似']
            
            if variation_index % 2 == 0:
                # 在中间添加修饰词
                mid_idx = len(words) // 2
                variation_word = variation_words[variation_index % len(variation_words)]
                words.insert(mid_idx, variation_word)
            else:
                # 添加过渡词
                if len(words) >= 3:
                    transition_word = transition_words[variation_index % len(transition_words)]
                    words.insert(-2, transition_word)
            
            return ''.join(words)
        except:
            return f"{text}（版本{variation_index + 1}）"
    
    def interactive_prediction(self):
        """交互式预测界面"""
        print("\n=== 增强歌词预测系统 ===")
        print("输入一句歌词，系统将为您预测可能的下一句")
        print("输入 'quit' 退出系统")
        print("-" * 50)
        
        while True:
            input_line = input("\n🎵 请输入一句歌词: ").strip()
            
            if input_line.lower() == 'quit':
                print("感谢使用歌词预测系统！")
                break
            
            if not input_line:
                print("请输入有效的歌词内容")
                continue
            
            print("\n🎯 正在预测...")
            predictions = self.predict_next_line_enhanced(input_line)
            
            print(f"\n📝 基于输入: \"{input_line}\"")
            print("🎼 预测的可能下一句:")
            
            for i, prediction in enumerate(predictions, 1):
                print(f"   {i}. {prediction}")
            
            print("-" * 50)


def main():
    """主函数"""
    predictor = EnhancedLyricsPredictor()
    
    print("=== 增强歌词预测系统 ===")
    print("1. 训练新的增强模型")
    print("2. 加载已有模型并开始预测")
    print("3. 仅收集歌词数据")
    
    choice = input("请选择操作 (1/2/3): ").strip()
    
    if choice == '1':
        print("开始训练增强模型...")
        predictor.train_enhanced_model(epochs=60)
        
        # 训练完成后进入交互模式
        if predictor.load_model():
            predictor.interactive_prediction()
    
    elif choice == '2':
        if predictor.load_model():
            predictor.interactive_prediction()
        else:
            print("模型加载失败！请先训练模型。")
    
    elif choice == '3':
        collector = LyricsCollector()
        lyrics = collector.expand_lyrics_dataset()
        print(f"数据收集完成，共收集 {len(lyrics)} 条歌词")
    
    else:
        print("无效选择！")


if __name__ == "__main__":
    main() 