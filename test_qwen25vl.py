"""Test script for Qwen2.5-VL model."""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# 辅助函数：加载图像
def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    return image

def main():
    # 模型路径
    model_path = "/workspace/Models/Qwen2.5-VL-7B-Instruct/"  # 请根据实际情况修改
    
    # 初始化模型和处理器
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()
    print("Model and processor loaded successfully!")
    
    # 测试纯文本输入
    print("\n=== Testing pure text input ===")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )
    
    output_text = processor.batch_decode(
        outputs[:, len(inputs.input_ids[0]):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"Input: Hello, how are you?")
    print(f"Output: {output_text}")
    
    # 测试多模态输入（需要有可用的图像URL）
    print("\n=== Testing multimodal input ===")
    try:
        # 使用一个公开的图像URL
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        image = load_image(image_url)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."}
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        
        output_text = processor.batch_decode(
            outputs[:, len(inputs.input_ids[0]):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Image URL: {image_url}")
        print(f"Input question: Describe this image.")
        print(f"Output: {output_text}")
        
    except Exception as e:
        print(f"Error during multimodal test: {e}")
        print("Please check the image URL or network connection.")

if __name__ == "__main__":
    main()
