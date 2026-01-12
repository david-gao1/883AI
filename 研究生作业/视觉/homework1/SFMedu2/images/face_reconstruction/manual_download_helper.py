#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视角人脸图片下载助手
"""

import os
import requests
import json

def download_from_url_list():
    """从URL列表下载"""
    output_dir = "images/face_reconstruction"
    os.makedirs(output_dir, exist_ok=True)
    
    print("请输入图片URL（每行一个，输入空行结束）：")
    urls = []
    while True:
        url = input().strip()
        if not url:
            break
        urls.append(url)
    
    if not urls:
        print("未输入任何URL")
        return
    
    print(f"\n开始下载 {len(urls)} 张图片...")
    for i, url in enumerate(urls, 1):
        filename = os.path.join(output_dir, f"face_{i:02d}.jpg")
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ {i}/{len(urls)}: face_{i:02d}.jpg")
            else:
                print(f"  ✗ {i}/{len(urls)}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ✗ {i}/{len(urls)}: {str(e)[:50]}")

if __name__ == "__main__":
    download_from_url_list()
