import os
import wandb

# 1. 手动指定你的API Key（也可以省略，用已登录的凭证）
# 替换成你从W&B官网获取的真实API Key
WANDB_API_KEY = "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"  
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

# 2. 核心测试逻辑
try:
    # 登录（验证API Key有效性）
    wandb.login()
    print("✅ W&B 登录成功！")
    
    # 初始化run（验证entity和project权限）
    run = wandb.init(
        project="qwen25vl",  # 你原代码中的project
        entity="1192445377-zhejiang-university", # 你原代码中的entity
        mode="online",       # 强制在线模式，暴露真实问题
        dir="./wandb_test"   # 临时目录
    )
    print("✅ W&B run 初始化成功！")
    
    # 上传测试数据（验证数据上传权限）
    run.log({"test_metric": 0.95})
    print("✅ 测试数据上传成功！")
    
    # 结束run（验证正常收尾）
    run.finish()
    print("🎉 所有W&B操作测试通过！")

except Exception as e:
    print(f"❌ 测试失败，错误原因：{e}")
    # 打印详细错误信息，方便定位
    import traceback
    traceback.print_exc()