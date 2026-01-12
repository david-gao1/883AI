#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速下载多视角人脸图片的工具
"""

import os
import sys
import requests
from pathlib import Path

def main():
    print("=" * 70)
    print("多视角人脸图片快速下载工具")
    print("=" * 70)
    print()
    print("请选择数据源：")
    print("1. 手动输入图片URL列表")
    print("2. 从本地文件夹选择图片")
    print("3. 查看下载指南")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        download_from_urls()
    elif choice == "2":
        select_from_folder()
    elif choice == "3":
        show_guide()
    else:
        print("无效选择")

def download_from_urls():
    output_dir = "images/face_reconstruction"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n请输入图片URL（每行一个，输入空行结束）：")
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

def select_from_folder():
    import glob
    import shutil
    
    source_dir = input("请输入包含图片的文件夹路径: ").strip()
    if not os.path.exists(source_dir):
        print("文件夹不存在")
        return
    
    output_dir = "images/face_reconstruction"
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找图片
    extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        images.extend(glob.glob(os.path.join(source_dir, ext)))
    
    if not images:
        print("未找到图片文件")
        return
    
    print(f"\n找到 {len(images)} 张图片")
    print("前10张图片：")
    for i, img in enumerate(images[:10], 1):
        print(f"  {i}. {os.path.basename(img)}")
    
    selected = input("\n请输入要使用的图片编号（用逗号分隔，例如: 1,2,3,4,5）: ").strip()
    indices = [int(x.strip()) - 1 for x in selected.split(',') if x.strip().isdigit()]
    
    for i, idx in enumerate(indices, 1):
        if 0 <= idx < len(images):
            src = images[idx]
            dst = os.path.join(output_dir, f"face_{i:02d}.jpg")
            shutil.copy2(src, dst)
            print(f"  ✓ 复制: {os.path.basename(src)} -> face_{i:02d}.jpg")
    
    print(f"\n完成！已准备 {len(indices)} 张图片")

def show_guide():
    print("
Recommended dataset download URLs:")
    print("
1. AFLW2000:")
    print("   http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm")
    print("
2. 300W-LP (PRNet):")
    print("   https://github.com/YadiraF/PRNet")
    print("
3. LFW:")
    print("   http://vis-www.cs.umass.edu/lfw/")
    print("
After downloading, use option 2 to select images from local folder.")

if __name__ == "__main__":
    main()
