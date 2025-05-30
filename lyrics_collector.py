import requests
import json
import time
import re
from bs4 import BeautifulSoup
import jieba
from urllib.parse import quote
import os


class LyricsCollector:
    """歌词收集器"""
    
    def __init__(self, output_file='lyrics_data.json'):
        self.output_file = output_file
        self.lyrics_data = []
        
    def clean_lyrics(self, lyrics):
        """清理歌词文本"""
        if not lyrics:
            return ""
        
        # 去除多余的空白字符
        lyrics = re.sub(r'\s+', ' ', lyrics)
        
        # 去除英文歌词标记
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        lyrics = re.sub(r'\(.*?\)', '', lyrics)
        
        # 去除特殊字符，保留中文、基本标点
        lyrics = re.sub(r'[^\u4e00-\u9fa5，。！？、；：""''（）\s]', '', lyrics)
        
        # 分割成行
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # 过滤太短的行
        lines = [line for line in lines if len(line) > 2]
        
        return lines
    
    def get_sample_chinese_lyrics(self):
        """获取一些中文歌词样本（避免版权问题，使用公开的古诗词和民歌）"""
        sample_lyrics = [
            # 古诗词改编
            "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
            "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
            "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
            "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
            
            # 民歌风格
            "茉莉花开满园香，采花姑娘心欢畅。",
            "山歌好比春江水，不怕滩险弯又多。",
            "阿里山的姑娘美如水，阿里山的少年壮如山。",
            "草原上升起不落的太阳，照耀着我们美丽的家乡。",
            "北风那个吹，雪花那个飘，雪花那个飘飘。",
            
            # 现代创作风格
            "春天来了花儿开，蝴蝶飞来又飞去。",
            "小河流水哗啦啦，鱼儿游来又游去。",
            "月亮弯弯照九州，几家欢乐几家愁。",
            "青山绿水好风光，鸟语花香满山冈。",
            "友谊如花永远开，真情如水永远流。",
            "梦想如星指引路，希望如光照前程。",
            "时光飞逝如流水，岁月如歌永流传。",
            "爱如春风暖人心，情如细雨润心田。",
            "勇敢面对每一天，微笑迎接新明天。",
            "心中有爱不孤单，眼中有光不迷茫。",
            
            # 励志类
            "不经历风雨，怎么见彩虹。",
            "阳光总在风雨后，乌云上有晴空。",
            "相信自己，相信梦想，明天会更好。",
            "勇敢飞翔，追逐梦想，青春无悔。",
            "坚持到底，永不放弃，成功就在前方。",
            
            # 抒情类
            "星空下的约定，永远不会忘记。",
            "微风吹过脸庞，带来温柔的思念。",
            "夜深人静的时候，想起远方的你。",
            "花开的季节里，想起美好的回忆。",
            "时光荏苒，友谊依然，感谢有你相伴。",
        ]
        
        return sample_lyrics
    
    def expand_lyrics_dataset(self):
        """扩展歌词数据集"""
        print("正在收集歌词数据...")
        
        # 获取基础样本
        basic_lyrics = self.get_sample_chinese_lyrics()
        
        processed_lyrics = []
        for lyric in basic_lyrics:
            lines = self.clean_lyrics(lyric)
            if lines:
                processed_lyrics.extend(lines)
        
        # 使用jieba分词创建更多的组合
        all_words = []
        for lyric in processed_lyrics:
            words = list(jieba.cut(lyric))
            all_words.extend(words)
        
        # 创建一些基于词频的新组合
        word_freq = {}
        for word in all_words:
            if len(word) > 1 and word not in ['，', '。', '！', '？']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 选择高频词创建新的句子组合
        common_words = [word for word, freq in word_freq.items() if freq >= 2]
        
        # 基于模板创建新的歌词
        templates = [
            "{} {} {}，{} {} {}。",
            "{} {} 如 {}，{} {} 满 {}。",
            "{} 的 {} 真美丽，{} 在 {} 里。",
            "{} {} 又 {}，{} {} 永不停。",
        ]
        
        generated_lyrics = []
        import random
        
        for template in templates:
            for _ in range(5):  # 每个模板生成5个例子
                try:
                    words_sample = random.sample(common_words, template.count('{}'))
                    new_lyric = template.format(*words_sample)
                    generated_lyrics.append(new_lyric)
                except:
                    continue
        
        all_lyrics = processed_lyrics + generated_lyrics
        
        # 保存到文件
        lyrics_data = {
            'lyrics': all_lyrics,
            'total_count': len(all_lyrics),
            'source': 'sample_and_generated'
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(lyrics_data, f, ensure_ascii=False, indent=2)
        
        print(f"收集完成！共收集 {len(all_lyrics)} 条歌词数据")
        print(f"数据已保存到 {self.output_file}")
        
        return all_lyrics
    
    def load_lyrics_data(self):
        """加载歌词数据"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('lyrics', [])
        except FileNotFoundError:
            print(f"数据文件 {self.output_file} 不存在，请先收集数据")
            return []
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return []


def main():
    """主函数"""
    collector = LyricsCollector()
    
    print("=== 歌词数据收集器 ===")
    print("1. 收集歌词数据")
    print("2. 查看已收集的数据")
    
    choice = input("请选择操作 (1/2): ").strip()
    
    if choice == '1':
        lyrics = collector.expand_lyrics_dataset()
        print("\n示例歌词:")
        for i, lyric in enumerate(lyrics[:10]):
            print(f"{i+1}. {lyric}")
    
    elif choice == '2':
        lyrics = collector.load_lyrics_data()
        if lyrics:
            print(f"已收集 {len(lyrics)} 条歌词数据")
            print("\n前10条歌词:")
            for i, lyric in enumerate(lyrics[:10]):
                print(f"{i+1}. {lyric}")
        else:
            print("没有找到歌词数据")
    
    else:
        print("无效选择！")


if __name__ == "__main__":
    main() 