from flask import Flask, render_template, request, jsonify
import os
import sys
import traceback
from optimized_lyrics_predictor import OptimizedLyricsPredictor

app = Flask(__name__)

# 全局预测器实例
predictor = None

def initialize_predictor():
    """初始化预测器"""
    global predictor
    try:
        print("开始初始化优化版预测器...")
        predictor = OptimizedLyricsPredictor()
        
        # 尝试加载已有模型
        if not predictor.load_model():
            print("没有找到已训练的模型，正在训练新的优化模型...")
            predictor.train_optimized_model(epochs=40)  # 训练优化模型
            predictor.load_model()
        print("优化版预测器初始化完成！")
    except Exception as e:
        print(f"初始化预测器时出错: {e}")
        print(f"错误详情: {traceback.format_exc()}")

@app.route('/')
def index():
    """主页"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"渲染主页时出错: {e}")
        return f"页面加载错误: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    global predictor
    
    if predictor is None:
        return jsonify({'error': '模型未初始化'}), 500
    
    data = request.get_json()
    input_lyrics = data.get('lyrics', '').strip()
    
    if not input_lyrics:
        return jsonify({'error': '请输入歌词内容'}), 400
    
    try:
        # 使用优化版预测器生成预测
        predictions = predictor.predict_with_quality_control(
            input_lyrics, 
            num_predictions=3
        )
        
        return jsonify({
            'input': input_lyrics,
            'predictions': predictions,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': f'预测时出错: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """训练模型接口"""
    global predictor
    
    try:
        if predictor is None:
            predictor = OptimizedLyricsPredictor()
        
        # 开始训练优化模型
        predictor.train_optimized_model(epochs=60)
        predictor.load_model()
        
        return jsonify({
            'message': '模型训练完成！',
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': f'训练时出错: {str(e)}'}), 500

@app.route('/status')
def status():
    """获取系统状态"""
    global predictor
    
    model_exists = os.path.exists('optimized_lyrics_model.pth')
    vocab_exists = os.path.exists('optimized_vocab.pkl')
    model_loaded = predictor is not None and predictor.model is not None
    
    return jsonify({
        'model_file_exists': model_exists,
        'vocab_file_exists': vocab_exists,
        'model_loaded': model_loaded,
        'ready': model_loaded
    })

if __name__ == '__main__':
    try:
        print("正在启动歌词预测Web应用...")
        print("首次启动可能需要一些时间来训练模型...")
        
        # 在另一个线程中初始化预测器，避免阻塞启动
        import threading
        init_thread = threading.Thread(target=initialize_predictor)
        init_thread.start()
        
        print("Web服务器启动中...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"启动Web应用时出错: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        input("按回车键退出...") 