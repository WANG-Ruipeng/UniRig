import numpy as np
import torch
import yaml
from box import Box
import os
import lightning as L
import trimesh
from src.data.raw_data import RawData

# 从 UniRig 的 src 目录导入所有需要的函数和类
from src.inference.download import download
from src.data.datapath import Datapath
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.transform import TransformConfig
from src.tokenizer.parse import get_tokenizer
from src.tokenizer.spec import TokenizerConfig
from src.model.parse import get_model
from src.system.parse import get_system

# --- 第2步 D部分：最终、最准确的推理脚本 ---

# 全局变量，用于被 hook 函数修改
hidden_state_from_hook = None

def get_mlp_input_hook(module, input, output):
    """PyTorch hook 函数，用于捕获指定网络层的输入。"""
    global hidden_state_from_hook
    print(f"✅ Hook 已被触发，捕获到目标层的输入，形状为: {input[0].shape}")
    hidden_state_from_hook = input[0].detach().cpu()

def load(name: str, path: str) -> Box:
    """
    完全模仿 run.py 中的 load 函数。
    """
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92m加载 {name} 配置: {path}\033[0m")
    return Box(yaml.safe_load(open(path, 'r')))

def preprocess_obj_to_npz(obj_path: str, output_dir: str) -> str:
    """
    手动将 .obj 文件预处理为 raw_data.npz 文件。
    这替代了 extract.py 中对 Blender 的依赖。

    Args:
        obj_path (str): 输入的 .obj 文件路径。
        output_dir (str): 用于存放 .npz 文件的目录。

    Returns:
        str: 生成的 .npz 文件的完整路径。
    """
    print(f"\n⚙️  开始手动预处理: {obj_path}")
    
    # 1. 使用 trimesh 加载 .obj 文件
    try:
        mesh = trimesh.load_mesh(obj_path)
    except Exception as e:
        print(f"❌ 使用 trimesh 加载 .obj 文件失败: {e}")
        return None

    # 2. 准备 RawData 对象所需的数据
    # 从我们最初的 .anime 文件转换过来的 .obj 只包含顶点和面片
    raw_data_instance = RawData(
        vertices=np.array(mesh.vertices, dtype=np.float32),
        faces=np.array(mesh.faces, dtype=np.int64),
        vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
        face_normals=np.array(mesh.face_normals, dtype=np.float32),
        joints=None,
        tails=None,
        skin=None,
        no_skin=None,
        parents=None,
        names=None,
        matrix_local=None,
    )
    
    # 3. 定义输出路径并保存
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, 'raw_data.npz')
    raw_data_instance.save(npz_path)
    
    print(f"✅ 成功生成预处理文件: {npz_path}")
    return npz_path

def run_unirig_inference_final(processed_data_dir: str):
    """
    最终修正版：严格按照 run.py 的逻辑，对单个 obj 文件进行推理。
    """
    global hidden_state_from_hook
    hidden_state_from_hook = None # 每次调用前重置

    # 1. 设置参数 (替代 argparse)
    # 使用您找到的正确的 task 配置文件！
    task_config_path = 'configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml'
    
    # 2. 加载配置并构建组件 (严格模仿 run.py)
    task = load('task', task_config_path)
    
    # --- 逐一加载所有组件配置 ---
    data_config = load('data', os.path.join('configs/data', task.components.data))
    transform_config = load('transform', os.path.join('configs/transform', task.components.transform))
    tokenizer_config_path = task.components.get('tokenizer', None)
    if tokenizer_config_path:
        tokenizer_config = load('tokenizer', os.path.join('configs/tokenizer', task.components.tokenizer))
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
    else:
        tokenizer_config = None
        
    predict_dataset_config = DatasetConfig.parse(config=data_config.predict_dataset_config).split_by_cls()
    predict_transform_config = TransformConfig.parse(config=transform_config.predict_transform_config)

    # --- 构建 DataModule ---
    datapath = Datapath(files=[processed_data_dir], cls=None)
    data_module = UniRigDatasetModule(
        process_fn=None,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        tokenizer_config=tokenizer_config,
        datapath=datapath,
    )
    
    # --- 构建 Model ---
    model_config = load('model', os.path.join('configs/model', task.components.model))
    tokenizer = get_tokenizer(config=tokenizer_config) if tokenizer_config else None
    model = get_model(tokenizer=tokenizer, **model_config)
    data_module.process_fn = model._process_fn # 关联 model 的处理函数到 datamodule
    
    # --- 构建 System ---
    system_config = load('system', os.path.join('configs/system', task.components.system))
    system = get_system(**system_config, model=model, steps_per_epoch=1)

    # 3. 【关键任务】找到目标层并注册 Hook
    # 这个逻辑不变，但现在我们操作的 model 对象是通过完全正确的方式构建的
    try:
        # ❗❗❗ 这是您需要做的最后一步侦查工作 ❗❗❗
        # 取消下面一行的注释来打印模型结构，然后找到正确的层
        # print(model) 
        
        target_layer = model.transformer.lm_head # <--- ！！！请根据打印出的结构，替换成真实的路径！！！
        print(f"✅ 成功找到目标层: {target_layer}")
        handle = target_layer.register_forward_hook(get_mlp_input_hook)
    except AttributeError as e:
        print(f"❌ [错误] 无法找到指定的目标层。请取消上面 `print(model)` 的注释，")
        print("   运行脚本，检查打印出的模型结构，然后更正 `target_layer` 的路径。")
        return None, None

    # 4. 执行推理 (使用 PyTorch Lightning Trainer)
    # 获取预训练模型权重路径并自动下载
    ckpt_path = task.get('resume_from_checkpoint', None)
    if ckpt_path:
        ckpt_path = download(ckpt_path)
    else:
        print("❌ [错误] 在 task 配置文件中未找到 `resume_from_checkpoint` 路径！")
        return None, None
    
    trainer_config = task.get('trainer', {})
    trainer = L.Trainer(**trainer_config)

    print(f"\n🚀 开始对 {processed_data_dir} 进行推理...")
    predictions = trainer.predict(system, datamodule=data_module, ckpt_path=ckpt_path)
    
    handle.remove() # 卸载 hook

    # 5. 提取并返回结果
    # 注意：`trainer.predict()` 的返回值可能为空列表 `[]`，因为结果通常由一个 "writer" 回调函数直接写入文件。
    # 但这没关系，我们的主要目标 `hidden_state_from_hook` 已经被 hook 函数捕获了。
    predicted_skeleton_obj = predictions
    
    if hidden_state_from_hook is None:
        print("⚠️ 警告: 推理完成，但 hook 未捕获到任何数据。请检查 target_layer 路径是否正确。")
        
    return predicted_skeleton_obj, hidden_state_from_hook

if __name__ == '__main__':
    # 确保脚本在 UniRig 项目根目录运行
    input_obj_file = '../BridgeForMotion/scripts/bear3EP_Agression_frame0.obj'
    
    # 为我们的 npz 文件创建一个临时输出目录
    # 路径结构模仿 UniRig 的预期：<output_dir>/<file_basename>/raw_data.npz
    file_basename = os.path.splitext(os.path.basename(input_obj_file))[0]
    temp_npz_dir = os.path.join('temp_preprocess', file_basename)

    if not os.path.exists(input_obj_file):
        print(f"❌ [错误] 输入的 .obj 文件不存在: {input_obj_file}")
    else:
        # 1. 在推理前，先执行预处理，生成 .npz 文件
        final_npz_path = preprocess_obj_to_npz(obj_path=input_obj_file, output_dir=temp_npz_dir)
        
        if final_npz_path:
            # 2. 将生成的 .npz 文件所在的【目录】传递给 UniRig 的推理函数
            # 这是关键！UniRig 的数据加载器需要的是包含 npz 的目录路径
            predicted_skeleton, hidden_state_X = run_unirig_inference_final(processed_data_dir=temp_npz_dir)
    
            if hidden_state_X is not None:
                print("\n" + "="*30)
                print("🎉 恭喜！整个流程已打通！ 🎉")
                print("="*30)
                print(f"✅ 捕获到的隐藏状态 (X) 的形状: {hidden_state_X.shape}")
                print(f"✅ 推理函数返回的骨架对象: {predicted_skeleton}")