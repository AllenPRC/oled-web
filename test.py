import os
import time
import base64

def encoder_image(image_path):
    """将图片转换为base64格式"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')
    

class PrintCollector:
    def __init__(self):
        self.logs = []  # 改为列表存储所有日志

    def log(self, msg):
        self.logs.append(msg)  # 追加日志而不是替换

    def run(self):
        images_path = os.listdir("/home/tju/deepseek/web/tmp")
        for i in range(1, 10):
            self.log(f"第 {i} 行输出, data:image/png;base64,{encoder_image(f'/home/tju/deepseek/web/tmp/{images_path[i]}')}")
            time.sleep(2)
