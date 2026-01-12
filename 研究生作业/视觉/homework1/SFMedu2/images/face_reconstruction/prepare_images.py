#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从公开数据集下载多视角人脸图片
使用方法：
1. 访问 AFLW2000: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
2. 下载数据集
3. 选择同一人的多张不同角度照片
4. 使用此脚本处理
"""

import os
import shutil
import glob

# 配置
source_dir = input("请输入下载的数据集文件夹路径: ").strip()
target_dir = "images/face_reconstruction"
os.makedirs(target_dir, exist_ok=True)

# 查找所有图片
image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
    image_files.extend(glob.glob(os.path.join(source_dir, ext)))

if not image_files:
    print(f"在 {source_dir} 中未找到图片文件")
    exit(1)

print(f"找到 {len(image_files)} 张图片")
print("选择要使用的图片（输入编号，用逗号分隔，例如: 1,2,3,4,5）:")
for i, img in enumerate(image_files[:20], 1):  # 只显示前20张
    print(f"  {i}. {os.path.basename(img)}")

selected = input("请输入选择的图片编号: ").strip().split(',')
selected_indices = [int(x.strip()) - 1 for x in selected if x.strip().isdigit()]

# 复制选中的图片
for i, idx in enumerate(selected_indices, 1):
    if 0 <= idx < len(image_files):
        src = image_files[idx]
        dst = os.path.join(target_dir, f"face_{i:02d}.jpg")
        shutil.copy2(src, dst)
        print(f"复制: {os.path.basename(src)} -> face_{i:02d}.jpg")

print(f"\n完成！已将 {len(selected_indices)} 张图片复制到 {target_dir}")
