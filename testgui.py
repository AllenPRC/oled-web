import os
os.environ['DISPLAY'] = ':0'
os.environ['XAUTHORITY'] = '/home/tju/.Xauthority'
import pyautogui



import requests
import base64
from PIL import Image
import io
import time

def encode_image(image_path):
    """将图片转换为base64格式"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_base64_image(base64_str, output_path):
    """将base64图片保存为文件"""
    img_data = base64.b64decode(base64_str)
    with open(output_path, 'wb') as f:
        f.write(img_data)

def parse_image(image_path, omniparser_url="http://localhost:8000",box_threshold=0.01,
    iou_threshold=0.1,
    use_paddleocr=True,):
    """调用omniparser服务解析图片"""
    try:
        # 检查服务是否在运行
        probe_response = requests.get(f"{omniparser_url}/probe/")
        if probe_response.status_code != 200:
            raise Exception("Omniparser service is not available")

        # 转换图片为base64
        image_base64 = encode_image(image_path)
        
        # 调用解析服务
        response = requests.post(
            f"{omniparser_url}/parse/",
            json={"base64_image": image_base64,
                  "params":
                      {
                        "box_threshold": box_threshold,
                        "iou_threshold": iou_threshold,
                        "use_paddleocr": use_paddleocr,
                      }
                      }
        )

        if response.status_code != 200:
            raise Exception(f"Parse failed with status code: {response.status_code}")
            
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to omniparser service: {e}")
        return None

def main():
    # 测试图片路径
    test_image_path = "/home/tju/deepseek/test/screenshots/10.1016%2Fj.apmt.2016.09.004/screenshot_20250417_025014.png"  # 替换为你的测试图片路径
    output_dir = "output"  # 输出目录
    
    # 确保输出目录存在
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting image parsing...")
    start_time = time.time()
    
    # 调用解析服务
    result = parse_image(test_image_path)
    
    if result:
        # 保存标注后的图片
        output_image_path = os.path.join(output_dir, "labeled_image.png")
        save_base64_image(result["som_image_base64"], output_image_path)
        
        # 打印解析结果
        print("\nParsed content:")
        for i, content in enumerate(result["parsed_content_list"]):
            print(f"Element {i+1}: {content}")
        print(result["parsed_content_list"])
        
        # 打印性能信息
        print(f"\nProcessing time: {result['latency']:.2f} seconds")
        print(f"Total time (including network): {time.time() - start_time:.2f} seconds")
        print(f"\nLabeled image saved to: {output_image_path}")
    else:
        print("Failed to parse image")


if not os.path.exists("./tmp"):
    os.makedirs("./tmp")
image_path = f"./tmp/screen_1.png"
pyautogui.screenshot(image_path)


result = parse_image(image_path)

output_path = f"./tmp/labeled_1.png"
save_base64_image(result["som_image_base64"], output_path)
print(result['parsed_content_list'])