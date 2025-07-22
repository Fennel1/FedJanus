#!/usr/bin/env python3
"""
Web服务部署脚本
Web Service Deployment Script
"""

import os
import sys
import argparse
import webbrowser
import time
from threading import Timer

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import WebApp

def open_browser(url, delay=2):
    """延迟打开浏览器"""
    def open_url():
        print(f"正在打开浏览器: {url}")
        webbrowser.open(url)
    
    Timer(delay, open_url).start()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='联邦学习结果可视化Web服务部署')
    parser.add_argument('--host', default='localhost', help='服务器主机地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口 (默认: 5000)')
    parser.add_argument('--results-dir', default='./results', help='结果文件目录 (默认: ./results)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    parser.add_argument('--threaded', action='store_true', default=True, help='启用多线程模式')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    if not os.path.exists(args.results_dir):
        print(f"创建结果目录: {args.results_dir}")
        os.makedirs(args.results_dir, exist_ok=True)
    
    # 创建WebApp实例
    web_app = WebApp(results_dir=args.results_dir)
    
    # 如果不是调试模式且不禁用浏览器，则自动打开浏览器
    if not args.debug and not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        open_browser(url)
    
    try:
        # 启动Web服务器
        web_app.run_server(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=args.threaded
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()