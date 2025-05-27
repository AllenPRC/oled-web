import requests
import zipfile
import io
import os
import time

# === 用户配置区域 ===
token = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI5MDMwMjc5OSIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0ODI2NTg5NCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTc4NjI5MTU2ODAiLCJvcGVuSWQiOm51bGwsInV1aWQiOiIyMDc0Zjg5Yi0zYjM4LTQ1YTgtYWUwYS02NDNmNTMwZTBmZmQiLCJlbWFpbCI6IiIsImV4cCI6MTc0OTQ3NTQ5NH0.R1BJVG6HjDzF5jYHOyOXHtGnETTxF2Rdw0MUt2Cr1gGg5cnAztMe1lV_KVZIpS-jZGsUtfj8-craURhznhr62Q"
file_path = "/home/tju/deepseek/minerU-api/pdf/s41467-024-55680-2.pdf"
base_output_dir = "/home/tju/deepseek/minerU-api/output"

# === 初始化变量 ===
file_name = os.path.basename(file_path)
data_id = os.path.splitext(file_name)[0]
output_dir = os.path.join(base_output_dir, data_id)
os.makedirs(output_dir, exist_ok=True)

# === 第一步：获取上传链接 ===
upload_api = 'https://mineru.net/api/v4/file-urls/batch'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
}
data = {
    "enable_formula": True,
    "language": "en",
    "enable_table": True,
    "files": [
        {"name": file_name, "is_ocr": True, "data_id": data_id}
    ]
}

print("申请上传链接...")
response = requests.post(upload_api, headers=headers, json=data)
response.raise_for_status()
res_json = response.json()
upload_url = res_json['data']['file_urls'][0]
batch_id = res_json['data']['batch_id']
print(f"获取上传链接成功，batch_id: {batch_id}")

# === 第二步：上传 PDF 文件 ===
print(f"上传文件: {file_path}")
with open(file_path, 'rb') as f:
    upload_response = requests.put(upload_url, data=f)
    if upload_response.status_code == 200:
        print("文件上传成功")
    else:
        raise Exception(f"文件上传失败，状态码: {upload_response.status_code}")

# === 第三步：轮询解析状态 ===
print("等待解析完成...")
status_url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
while True:
    time.sleep(6)
    check = requests.get(status_url, headers=headers)
    result = check.json()
    extract_result = result["data"]["extract_result"][0]
    state = extract_result["state"]

    if state == "done":
        print("解析完成")
        zip_url = extract_result["full_zip_url"]
        break
    elif state == "failed":
        raise Exception(f"解析失败: {extract_result['err_msg']}")
    else:
        print(f"当前状态: {state}，继续等待...")

# === 第四步：下载 ZIP 文件并保存 ===
zip_response = requests.get(zip_url)
local_zip_path = os.path.join(output_dir, f"{data_id}.zip")
with open(local_zip_path, "wb") as f:
    f.write(zip_response.content)
print(f"ZIP 文件已保存至：{local_zip_path}")

# === 第五步：解压 ZIP 到 data_id 目录，并保存 Markdown 文件 ===
with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
    z.extractall(output_dir)
    print(f"已解压至目录：{output_dir}")

    md_found = False
    for name in z.namelist():
        if name.endswith(".md"):
            md_path = os.path.join(output_dir, f"{data_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(z.read(name).decode("utf-8"))
            print(f"Markdown 文件已保存至：{md_path}")
            md_found = True

    if not md_found:
        print("⚠️ 未在 ZIP 中找到 Markdown 文件。")
