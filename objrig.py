# -*- coding: utf-8 -*-
"""
一个独立的 Python 脚本，用于封装 UniRig 的骨架预测功能，
并提取一个中间层的隐藏状态。
此版本基于项目 `run.py` 的真实逻辑编写。
"""
# =================================================================
# === 步骤 1: 基础设置，完全模仿 run.py ===
import sys
import os
import yaml
from box import Box  # run.py 使用了 box 库来方便地访问配置项

# 将项目根目录动态添加到 Python 搜索路径
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# =================================================================

import torch
import pytorch_lightning as pl

# =================================================================
# === 步骤 2: 从正确的模块导入所有需要的函数和类 ===
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.datapath import Datapath
from src.data.transform import TransformConfig
from src.tokenizer.spec import TokenizerConfig, DetokenizeOutput
from src.tokenizer.parse import get_tokenizer
from src.model.parse import get_model
from src.system.parse import get_system
from src.inference.download import download
# =================================================================

# 全局变量，用于存储 hook 捕获的张量
captured_tensor = None

def load_config_from_yaml(path: str) -> Box:
    """模仿 run.py 的 load 函数，用于加载 yaml 配置文件。"""
    if not path.endswith('.yaml'):
        path += '.yaml'
    print(f"Loading config: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    return Box(yaml.safe_load(open(path, 'r')))

def forward_hook(module: torch.nn.Module, input_tensor: tuple[torch.Tensor], output_tensor: torch.Tensor):
    """一个前向 hook 函数，用于捕获模块的输入张量。"""
    global captured_tensor
    print("Hook triggered! Capturing the input tensor.")
    captured_tensor = input_tensor[0].detach().cpu()

def get_unirig_prediction(obj_path: str) -> tuple[DetokenizeOutput, torch.Tensor]:
    """
    封装 UniRig 的骨架预测流程，返回预测结果和一个中间层张量。
    """
    global captured_tensor
    captured_tensor = None

    # --- 设备处理 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =================================================================
    # === 步骤 3: 完全模仿 run.py 的配置加载流程 ===
    
    # 3.1. 加载各个部分的配置
    # 这些路径是基于 UniRig 项目的标准文件结构
    print("--- Loading Configurations ---")
    data_config = load_config_from_yaml('configs/data/quick_inference.yaml')
    transform_config = load_config_from_yaml('configs/transform/inference_ar_transform.yaml')
    model_config = load_config_from_yaml('configs/model/unirig_ar_350m_1024_81920_float32.yaml')
    system_config = load_config_from_yaml('configs/system/ar_inference_articulationxl.yaml')
    
    # 我们需要 tokenizer 的配置来创建模型
    tokenizer_config = load_config_from_yaml('configs/tokenizer/tokenizer_rignet.yaml')
    tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
    # 从 task.yaml 中获取 checkpoint 路径
    # 这里我们直接硬编码，因为它对于推理是固定的
    resume_from_checkpoint = 'https://huggingface.co/VAST-AI/UniRig/resolve/main/unirig-v1.ckpt'
    # =================================================================
    
    # --- 步骤 4: 模仿 run.py 创建对象 ---
    print("\n--- Instantiating Components ---")
    
    # 4.1 创建 tokenizer 和 model
    tokenizer = get_tokenizer(config=tokenizer_config) if tokenizer_config else None
    model = get_model(tokenizer=tokenizer, **model_config)

    # 4.2 创建数据模块 (DataModule)
    # 这是之前缺失的关键一步。我们需要它来为 trainer 提供数据。
    datapath = Datapath(files=[obj_path], cls=system_config.generate_kwargs.assign_cls)
    
    data = UniRigDatasetModule(
        process_fn=model._process_fn,
        predict_dataset_config=DatasetConfig.parse(config=data_config.predict_dataset_config),
        predict_transform_config=TransformConfig.parse(config=transform_config.predict_transform_config),
        tokenizer_config=tokenizer_config,
        data_name='raw_data.npz', # 使用默认值
        datapath=datapath,
        cls=system_config.generate_kwargs.assign_cls,
    )

    # 4.3 创建 system 对象
    system = get_system(
        **system_config,
        model=model,
        steps_per_epoch=1
    )
    
    system = system.to(device)
    system.eval()

    # 4.4 创建 Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    # --- 步骤 5: 注册 Hook 并执行推理 ---
    print("\n--- Starting Prediction ---")
    hook_handle = None
    try:
        target_layer = system.model.to_logits
        print(f"Registering a forward hook on layer: system.model.to_logits")
        hook_handle = target_layer.register_forward_hook(forward_hook)

        print("Downloading checkpoint if necessary...")
        ckpt_path = download(resume_from_checkpoint)
        
        print("Running trainer.predict...")
        prediction_output = trainer.predict(system, datamodule=data, ckpt_path=ckpt_path)[0]
        final_prediction = prediction_output[0]
        print("Prediction finished.")

    finally:
        if hook_handle:
            hook_handle.remove()
            print("Forward hook has been removed.")

    if captured_tensor is None:
        raise RuntimeError("Hook was not triggered. Failed to capture the tensor.")
    print("Successfully captured the hidden state tensor.")

    return final_prediction, captured_tensor


if __name__ == '__main__':
    user_obj_path = r"D:\Words\CIS900\MotionGen\BridgeForMotion\scripts\bear3EP_Agression_frame0.obj"

    if not os.path.exists(user_obj_path):
        print(f"错误：找不到文件 '{user_obj_path}'。请确保路径正确，并且脚本有权访问该文件。")
    else:
        # 安装 PyYAML 和 aibox
        try:
            import yaml
            from box import Box
        except ImportError:
            print("错误：缺少必要的库。请运行 'pip install pyyaml python-box'")
            sys.exit(1)
            
        print(f"Target .obj file: {user_obj_path}")
        
        try:
            predicted_skeleton, hidden_tensor = get_unirig_prediction(obj_path=user_obj_path)
            
            print("\n" + "="*50)
            print("      UniRig Prediction and Hooking Result")
            print("="*50)
            print(f"\n[1] Predicted Skeleton Object Type: {type(predicted_skeleton)}")
            if isinstance(predicted_skeleton, DetokenizeOutput):
                print(f"    - Skeleton has {len(predicted_skeleton.joints)} joints.")
                print(f"    - Skeleton has {len(predicted_skeleton.edges)} edges.")
            
            print(f"\n[2] Captured Hidden State Tensor Type: {type(hidden_tensor)}")
            print(f"    - Tensor shape: {hidden_tensor.shape}")
            print(f"    - Tensor device: {hidden_tensor.device}")
            print(f"    - Tensor mean value: {hidden_tensor.float().mean().item()}")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()