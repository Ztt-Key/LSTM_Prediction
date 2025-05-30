#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
歌词预测系统 - 启动脚本
快速启动歌词预测系统的不同模式
"""

import os
import sys

def print_banner():
    """打印启动横幅"""
    banner = """
    🎵 ═══════════════════════════════════════════════════════════ 🎵
    
              🎤 欢迎使用歌词预测系统 🎤
              
        基于深度学习的中文歌词生成AI助手
        输入一句歌词，AI为您创作下一句！
        
    🎵 ═══════════════════════════════════════════════════════════ 🎵
    """
    print(banner)

def check_dependencies():
    """检查基本依赖"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import jieba
    except ImportError:
        missing_deps.append("jieba")
    
    try:
        import flask
    except ImportError:
        missing_deps.append("flask")
    
    if missing_deps:
        print("❌ 缺少以下依赖包:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 请运行以下命令安装:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True

def start_web_app():
    """启动Web应用"""
    print("🌐 启动Web界面...")
    print("📍 服务地址: http://localhost:5000")
    print("⏳ 首次启动需要训练模型，请耐心等待...")
    print("\n按 Ctrl+C 停止服务\n")
    
    try:
        # 导入并启动web应用
        import sys
        import os
        
        # 执行web_app.py
        os.system("py -3.11 web_app.py")
        
    except KeyboardInterrupt:
        print("\n👋 感谢使用歌词预测系统！")
    except Exception as e:
        print(f"❌ 启动Web应用失败: {e}")

def start_enhanced_cli():
    """启动增强版命令行界面"""
    print("💻 启动增强版命令行界面...")
    print("⏳ 正在初始化系统...")
    
    try:
        from enhanced_lyrics_predictor import main
        main()
    except KeyboardInterrupt:
        print("\n👋 感谢使用歌词预测系统！")
    except Exception as e:
        print(f"❌ 启动增强版失败: {e}")

def start_basic_cli():
    """启动基础版命令行界面"""
    print("💻 启动基础版命令行界面...")
    print("⏳ 正在初始化系统...")
    
    try:
        from lyrics_predictor import main
        main()
    except KeyboardInterrupt:
        print("\n👋 感谢使用歌词预测系统！")
    except Exception as e:
        print(f"❌ 启动基础版失败: {e}")

def collect_data():
    """收集歌词数据"""
    print("📊 启动数据收集器...")
    
    try:
        from lyrics_collector import main
        main()
    except KeyboardInterrupt:
        print("\n👋 数据收集已停止！")
    except Exception as e:
        print(f"❌ 数据收集失败: {e}")

def run_tests():
    """运行系统测试"""
    print("🔍 运行系统测试...")
    
    try:
        # 简单的系统测试
        print("✅ 正在检查核心模块...")
        
        # 测试基础预测器
        from lyrics_predictor import LyricsPredictor
        print("✅ 基础预测器模块正常")
        
        # 测试增强预测器
        from enhanced_lyrics_predictor import EnhancedLyricsPredictor
        print("✅ 增强预测器模块正常")
        
        # 测试数据收集器
        from lyrics_collector import LyricsCollector
        print("✅ 数据收集器模块正常")
        
        print("✅ 所有核心模块测试通过！")
        
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")

def show_help():
    """显示帮助信息"""
    help_text = """
    📖 使用说明:
    
    🌐 Web界面 (推荐)
       - 现代化的网页界面
       - 支持多设备访问
       - 实时状态显示
       - 美观的用户体验
    
    💻 命令行界面
       - 增强版: 更好的模型和预测能力
       - 基础版: 轻量级快速体验
       - 适合开发者和高级用户
    
    📊 数据收集
       - 扩展训练数据集
       - 提升模型性能
       - 自定义数据源
    
    🔍 系统测试
       - 验证环境配置
       - 检查依赖安装
       - 功能完整性测试
    
    💡 小贴士:
       - 首次使用需要训练模型（约3-5分钟）
       - Web界面最适合日常使用
       - 输入完整的歌词效果更好
       - 可以多次预测获得不同结果
       
    🚀 推荐使用方式:
       - 新手用户: 选择 "Web界面"
       - 开发者: 选择 "增强版命令行界面"
       - 快速体验: 直接运行 "py -3.11 optimized_lyrics_predictor.py"
    """
    print(help_text)

def main():
    """主菜单"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装必要的依赖包。")
        return
    
    while True:
        print("\n🚀 请选择启动方式:")
        print("   1. 🌐 Web界面 (推荐)")
        print("   2. 💻 增强版命令行界面")
        print("   3. 💻 基础版命令行界面")
        print("   4. 📊 数据收集器")
        print("   5. 🔍 运行系统测试")
        print("   6. 📖 查看帮助")
        print("   0. 👋 退出程序")
        
        choice = input("\n请输入选择 (0-6): ").strip()
        
        if choice == "1":
            start_web_app()
            break
        elif choice == "2":
            start_enhanced_cli()
            break
        elif choice == "3":
            start_basic_cli()
            break
        elif choice == "4":
            collect_data()
        elif choice == "5":
            run_tests()
        elif choice == "6":
            show_help()
        elif choice == "0":
            print("\n👋 感谢使用歌词预测系统！")
            break
        else:
            print("❌ 无效选择，请重新输入。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 感谢使用歌词预测系统！")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        print("请检查环境配置或联系开发者。") 