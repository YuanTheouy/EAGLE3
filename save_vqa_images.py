#!/usr/bin/env python3
"""
保存VQA数据集图片的工具脚本
"""

import os
import json
import requests
from PIL import Image
from io import BytesIO
import argparse


def load_questions(question_file):
    """加载VQA问题文件"""
    questions = []
    with open(question_file, "r") as fin:
        for line in fin:
            questions.append(json.loads(line))
    return questions


def load_image(image_path_or_url):
    """加载图片（支持本地路径和URL）"""
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        # 检查路径是否存在
        if not os.path.exists(image_path_or_url):
            print(f"警告：图片路径不存在: {image_path_or_url}")
            return None
        image = Image.open(image_path_or_url)
    return image


def save_vqa_images(question_file, output_dir, max_images=20):
    """保存VQA数据集中的图片"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载问题
    questions = load_questions(question_file)
    print(f"总共加载了 {len(questions)} 个问题")
    
    # 限制处理的图片数量
    if max_images > 0:
        questions = questions[:max_images]
        print(f"将保存前 {max_images} 张图片")
    
    # 保存图片
    saved_count = 0
    for i, question in enumerate(questions):
        try:
            # 检查问题是否包含图像字段
            if "image" not in question:
                print(f"问题 {i} 没有图像字段")
                continue
            
            image_url = question["image"]
            print(f"处理问题 {i}: {image_url}")
            
            # 加载图片
            image = load_image(image_url)
            if image is None:
                continue
            
            # 生成保存文件名
            # 从问题ID生成文件名
            question_id = question.get("question_id", i)
            # 获取图片扩展名
            if image_url.endswith('.jpg') or image_url.endswith('.jpeg'):
                ext = '.jpg'
            elif image_url.endswith('.png'):
                ext = '.png'
            elif image_url.endswith('.gif'):
                ext = '.gif'
            else:
                ext = '.jpg'  # 默认使用jpg
            
            # 保存图片
            save_path = os.path.join(output_dir, f"image_{question_id}{ext}")
            image.save(save_path)
            print(f"已保存: {save_path}")
            
            saved_count += 1
            
        except Exception as e:
            print(f"处理问题 {i} 时出错: {e}")
            continue
    
    print(f"\n处理完成，共保存了 {saved_count} 张图片到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存VQA数据集图片的工具脚本")
    parser.add_argument("--question-file", required=True, help="VQA问题文件路径")
    parser.add_argument("--output-dir", default="vqa_images", help="图片输出目录")
    parser.add_argument("--max-images", type=int, default=20, help="最大保存图片数量")
    
    args = parser.parse_args()
    
    save_vqa_images(args.question_file, args.output_dir, args.max_images)