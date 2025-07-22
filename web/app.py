"""
Flask Web应用 - 联邦学习结果可视化
Flask Web Application - Federated Learning Results Visualization
"""

import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
import base64

class WebApp:
    def __init__(self, results_dir="./results"):
        self.app = Flask(__name__, static_folder='static', static_url_path='/static')
        self.app.config['SECRET_KEY'] = 'federated_learning_web_app'
        self.app.config['UPLOAD_FOLDER'] = results_dir
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.results_dir = results_dir
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 确保静态文件目录存在
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
        
        # 注册路由和错误处理
        self._register_routes()
        self._register_error_handlers()
    
    def _register_routes(self):
        """注册Flask路由"""
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/upload', 'upload_file', self.upload_file, methods=['POST'])
        self.app.add_url_rule('/visualize/<filename>', 'visualize', self.visualize)
        self.app.add_url_rule('/api/results/<filename>', 'get_results', self.get_results)
        self.app.add_url_rule('/api/health', 'health_check', self.health_check)
        self.app.add_url_rule('/api/files', 'list_files', self.list_files)
    
    def _register_error_handlers(self):
        """注册错误处理器"""
        @self.app.errorhandler(404)
        def not_found_error(error):
            return render_template('error.html', 
                                 error_message="页面不存在 (404)"), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return render_template('error.html', 
                                 error_message="服务器内部错误 (500)"), 500
        
        @self.app.errorhandler(413)
        def file_too_large(error):
            return render_template('error.html', 
                                 error_message="文件过大，请选择小于16MB的文件"), 413
        
        @self.app.errorhandler(Exception)
        def handle_exception(error):
            # 记录错误日志
            self.app.logger.error(f"未处理的异常: {str(error)}")
            return render_template('error.html', 
                                 error_message=f"系统错误: {str(error)}"), 500
    
    def index(self):
        """主页 - 显示结果文件选择界面"""
        # 获取结果目录中的所有JSON文件
        result_files = []
        if os.path.exists(self.results_dir):
            for filename in os.listdir(self.results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.results_dir, filename)
                    try:
                        # 尝试读取文件获取实验信息
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            experiment_info = data.get('experiment_info', {})
                            result_files.append({
                                'filename': filename,
                                'dataset': experiment_info.get('dataset', 'Unknown'),
                                'model': experiment_info.get('model', 'Unknown'),
                                'num_clients': experiment_info.get('num_clients', 'Unknown'),
                                'attack_type': experiment_info.get('attack_type', 'None')
                            })
                    except (json.JSONDecodeError, IOError):
                        # 如果文件无法读取，仍然显示文件名
                        result_files.append({
                            'filename': filename,
                            'dataset': 'Error',
                            'model': 'Error',
                            'num_clients': 'Error',
                            'attack_type': 'Error'
                        })
        
        return render_template('index.html', result_files=result_files)
    
    def upload_file(self):
        """处理文件上传"""
        if 'file' not in request.files:
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('index'))
        
        if file and file.filename.endswith('.json'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                # 验证JSON文件格式
                with open(filepath, 'r', encoding='utf-8') as f:
                    json.load(f)
                return redirect(url_for('visualize', filename=filename))
            except (json.JSONDecodeError, IOError) as e:
                # 删除无效文件
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template('error.html', 
                                     error_message=f"文件格式错误: {str(e)}")
        
        return redirect(url_for('index'))
    
    def visualize(self, filename):
        """可视化页面"""
        filepath = os.path.join(self.results_dir, filename)
        if not os.path.exists(filepath):
            return render_template('error.html', 
                                 error_message="文件不存在")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return render_template('visualize.html', 
                                 filename=filename, 
                                 experiment_info=data.get('experiment_info', {}))
        except (json.JSONDecodeError, IOError) as e:
            return render_template('error.html', 
                                 error_message=f"文件读取错误: {str(e)}")
    
    def get_results(self, filename):
        """API接口 - 获取结果数据"""
        filepath = os.path.join(self.results_dir, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify(data)
        except (json.JSONDecodeError, IOError) as e:
            return jsonify({'error': f'文件读取错误: {str(e)}'}), 500
    
    def health_check(self):
        """健康检查API"""
        return jsonify({
            'status': 'healthy',
            'service': 'Federated Learning Visualization',
            'version': '1.0.0',
            'results_dir': self.results_dir,
            'available_files': len([f for f in os.listdir(self.results_dir) 
                                  if f.endswith('.json')]) if os.path.exists(self.results_dir) else 0
        })
    
    def list_files(self):
        """API接口 - 获取所有结果文件列表"""
        try:
            result_files = []
            if os.path.exists(self.results_dir):
                for filename in os.listdir(self.results_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.results_dir, filename)
                        try:
                            # 获取文件信息
                            file_stat = os.stat(filepath)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                experiment_info = data.get('experiment_info', {})
                            
                            result_files.append({
                                'filename': filename,
                                'size': file_stat.st_size,
                                'modified': file_stat.st_mtime,
                                'experiment_info': experiment_info
                            })
                        except (json.JSONDecodeError, IOError, OSError):
                            # 如果文件无法读取，仍然包含基本信息
                            result_files.append({
                                'filename': filename,
                                'size': 0,
                                'modified': 0,
                                'experiment_info': {'error': 'File read error'}
                            })
            
            return jsonify({
                'files': result_files,
                'total': len(result_files)
            })
        except Exception as e:
            return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500
    
    def run_server(self, host='localhost', port=5000, debug=False, threaded=True):
        """启动Web服务器"""
        import logging
        from werkzeug.serving import WSGIRequestHandler
        
        # 配置日志
        if not debug:
            logging.basicConfig(level=logging.INFO)
            self.app.logger.setLevel(logging.INFO)
        
        # 检查端口是否可用
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"警告: 端口 {port} 已被占用，尝试使用其他端口...")
            for test_port in range(port + 1, port + 10):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((host, test_port))
                sock.close()
                if result != 0:
                    port = test_port
                    break
            else:
                raise RuntimeError("无法找到可用端口")
        
        print("=" * 60)
        print("联邦学习结果可视化系统")
        print("=" * 60)
        print(f"服务器地址: http://{host}:{port}")
        print(f"结果目录: {os.path.abspath(self.results_dir)}")
        print(f"调试模式: {'开启' if debug else '关闭'}")
        print(f"多线程: {'开启' if threaded else '关闭'}")
        
        # 显示可用的结果文件
        if os.path.exists(self.results_dir):
            json_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
            if json_files:
                print(f"发现 {len(json_files)} 个结果文件:")
                for f in json_files[:5]:  # 只显示前5个
                    print(f"  - {f}")
                if len(json_files) > 5:
                    print(f"  ... 还有 {len(json_files) - 5} 个文件")
            else:
                print("结果目录中暂无JSON文件")
        else:
            print("结果目录不存在，将自动创建")
        
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        try:
            # 自定义请求处理器以支持更好的日志
            class CustomRequestHandler(WSGIRequestHandler):
                def log_request(self, code='-', size='-'):
                    if not debug:  # 只在非调试模式下记录访问日志
                        self.log('info', '"%s" %s %s', self.requestline, code, size)
            
            self.app.run(
                host=host, 
                port=port, 
                debug=debug, 
                threaded=threaded,
                request_handler=CustomRequestHandler if not debug else None
            )
        except KeyboardInterrupt:
            print("\n服务器已停止")
        except Exception as e:
            print(f"服务器启动失败: {str(e)}")
            raise

# 创建全局应用实例
web_app = WebApp()

if __name__ == '__main__':
    web_app.run_server(debug=True)