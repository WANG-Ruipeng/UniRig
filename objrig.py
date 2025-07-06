import numpy as np
import torch
import yaml
from box import Box
import os
import lightning as L
import trimesh

# --- 新增的导入 ---
from src.tokenizer.spec import DetokenizeOutput 
# --- 结束新增 ---

from src.data.raw_data import RawData
from src.inference.download import download
from src.data.datapath import Datapath
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.transform import TransformConfig
from src.tokenizer.parse import get_tokenizer
from src.tokenizer.spec import TokenizerConfig
from src.model.parse import get_model
from src.system.parse import get_system

# 全局变量，用于被 hook 函数修改
hidden_state_from_hook = None

def get_mlp_input_hook(module, input, output):
    """PyTorch hook 函数，用于捕获指定网络层的输入。"""
    global hidden_state_from_hook
    # print(f"✅ Hook 已被触发，捕获到目标层的输入，形状为: {input[0].shape}")
    hidden_state_from_hook = input[0].detach().cpu()

def load(name: str, path: str) -> Box:
    """完全模仿 run.py 中的 load 函数。"""
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    # print(f"\033[92m加载 {name} 配置: {path}\033[0m")
    return Box(yaml.safe_load(open(path, 'r')))

def preprocess_obj_to_npz(obj_path: str, output_dir: str) -> str:
    """手动将 .obj 文件预处理为 raw_data.npz 文件。"""
    print(f"\n⚙️  开始手动预处理: {obj_path}")
    try:
        mesh = trimesh.load_mesh(obj_path)
    except Exception as e:
        print(f"❌ 使用 trimesh 加载 .obj 文件失败: {e}")
        return None

    raw_data_instance = RawData(
        vertices=np.array(mesh.vertices, dtype=np.float32),
        faces=np.array(mesh.faces, dtype=np.int64),
        vertex_normals=np.array(mesh.vertex_normals, dtype=np.float32),
        face_normals=np.array(mesh.face_normals, dtype=np.float32),
        joints=None, tails=None, skin=None, no_skin=None,
        parents=None, names=None, matrix_local=None,
    )
    
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, 'raw_data.npz')
    raw_data_instance.save(npz_path)
    
    print(f"✅ 成功生成预处理文件: {npz_path}")
    return npz_path

# =====================================================================
# --- 最终实现的 BVH 保存函数 ---
# =====================================================================
def save_skeleton_to_bvh(skeleton_output: DetokenizeOutput, bvh_path: str):
    """
    将 UniRig 的骨架输出对象 (DetokenizeOutput) 保存为 .bvh 文件。

    Args:
        skeleton_output: 从 UniRig 推理得到的 DetokenizeOutput 对象。
        bvh_path (str): 输出的 .bvh 文件路径。
    """
    print(f"\n💾  开始将骨骼保存为 .bvh 文件: {bvh_path}...")
    
    # 1. 从 DetokenizeOutput 对象中安全地提取数据
    try:
        # 使用 .joints 属性获取关节位置
        joints_pos = skeleton_output.joints
        if isinstance(joints_pos, torch.Tensor):
            joints_pos = joints_pos.cpu().numpy()

        # 使用 ._get_parents() 方法获取最可靠的父子关系
        parents_idx = skeleton_output._get_parents()
        
        # 使用 .names 属性，如果不存在则创建默认名字
        joint_names = skeleton_output.names
        if joint_names is None or len(joint_names) != len(joints_pos):
            print("⚠️  警告: 找不到关节名称或数量不匹配，将使用默认名称 'joint_N'。")
            joint_names = [f"joint_{i}" for i in range(len(joints_pos))]
            
        num_joints = joints_pos.shape[0]

    except Exception as e:
        print(f"❌ 错误：从骨架对象中提取数据失败。错误信息: {e}")
        return

    # 2. 构建写入 .bvh 所需的父子关系树
    children = [[] for _ in range(num_joints)]
    root_idx = -1
    for i, p_idx in enumerate(parents_idx):
        if p_idx is None: # 根节点的父节点是 None
            root_idx = i
        else:
            children[p_idx].append(i)

    if root_idx == -1 and len(joints_pos) > 0:
        print("⚠️  警告: 找不到根节点，将假定第一个关节为根节点。")
        root_idx = 0
    elif len(joints_pos) == 0:
        print("❌ 错误: 骨架中没有任何关节。")
        return

    # 3. 开始写入文件
    with open(bvh_path, 'w') as f:
        f.write("HIERARCHY\n")

        def write_joint_hierarchy(joint_idx, indent_level):
            indent = "  " * indent_level
            is_root = (parents_idx[joint_idx] is None)
            
            f.write(f"{indent}ROOT {joint_names[joint_idx]}\n" if is_root else f"{indent}JOINT {joint_names[joint_idx]}\n")
            f.write(f"{indent}{{\n")

            if is_root:
                offset = joints_pos[joint_idx]
            else:
                parent_pos = joints_pos[parents_idx[joint_idx]]
                offset = joints_pos[joint_idx] - parent_pos
            
            f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
            f.write(f"{indent}  CHANNELS 3 Zrotation Yrotation Xrotation\n")

            if not children[joint_idx]: # 末端关节
                f.write(f"{indent}  End Site\n")
                f.write(f"{indent}  {{\n")
                f.write(f"{indent}    OFFSET 0.0 0.0 0.0\n")
                f.write(f"{indent}  }}\n")
            else:
                for child_idx in children[joint_idx]:
                    write_joint_hierarchy(child_idx, indent_level + 1)
            
            f.write(f"{indent}}}\n")

        write_joint_hierarchy(root_idx, 0)

        f.write("\nMOTION\n")
        f.write(f"Frames: 1\n")
        f.write("Frame Time: 0.033333\n")
        motion_data = " ".join(["0.0"] * num_joints * 3)
        f.write(motion_data + "\n")

    print(f"✅ 成功保存 .bvh 文件: {bvh_path}")


def run_unirig_inference_final(processed_data_dir: str):
    """最终修正版：读取预处理好的 .npz 文件并进行推理。"""
    # 此函数内容无需修改，保持原样即可
    global hidden_state_from_hook
    hidden_state_from_hook = None
    task_config_path = 'configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml'
    task = load('task', task_config_path)
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
    datapath = Datapath(files=[processed_data_dir], cls=None)
    data_module = UniRigDatasetModule(
        process_fn=None, predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config, tokenizer_config=tokenizer_config,
        datapath=datapath,
    )
    model_config = load('model', os.path.join('configs/model', task.components.model))
    tokenizer = get_tokenizer(config=tokenizer_config) if tokenizer_config else None
    model = get_model(tokenizer=tokenizer, **model_config)
    data_module.process_fn = model._process_fn
    system_config = load('system', os.path.join('configs/system', task.components.system))
    system = get_system(**system_config, model=model, steps_per_epoch=1)
    
    try:
        target_layer = model.transformer.lm_head
        handle = target_layer.register_forward_hook(get_mlp_input_hook)
    except AttributeError:
        # 此处错误处理已不再需要，但保留以防万一
        return None, None
    
    ckpt_path = download(task.get('resume_from_checkpoint', None))
    trainer_config = task.get('trainer', {})
    trainer = L.Trainer(**trainer_config)
    
    print(f"\n🚀 开始对 {processed_data_dir} 进行推理...")
    predictions = trainer.predict(system, datamodule=data_module, ckpt_path=ckpt_path)
    
    handle.remove()
    return predictions, hidden_state_from_hook

# =====================================================================
# --- 主执行函数 ---
# =====================================================================
if __name__ == '__main__':
    # 确保脚本在 UniRig 项目根目录运行
    input_obj_file = '../BridgeForMotion/scripts/bear3EP_Agression_frame0.obj'
    
    # 从输入路径中获取基本信息
    file_basename = os.path.splitext(os.path.basename(input_obj_file))[0]
    input_dir = os.path.dirname(input_obj_file)
    
    # 定义中间文件的输出目录
    temp_npz_dir = os.path.join('temp_preprocess', file_basename)
    
    # 定义最终的 BVH 输出路径，使其与输入文件在同一目录下
    output_bvh_path = os.path.join(input_dir, f"{file_basename}_rig.bvh")

    if not os.path.exists(input_obj_file):
        print(f"❌ [错误] 输入的 .obj 文件不存在: {input_obj_file}")
    else:
        # 步骤 1: 预处理
        final_npz_path = preprocess_obj_to_npz(obj_path=input_obj_file, output_dir=temp_npz_dir)
        
        if final_npz_path:
            # 步骤 2: UniRig 推理
            predictions, hidden_state_X = run_unirig_inference_final(processed_data_dir=temp_npz_dir)
    
            # 步骤 3: 检查结果并保存 BVH
            if hidden_state_X is not None and predictions and len(predictions) > 0 and len(predictions[0]) > 0:
                print("\n" + "="*30)
                print("🎉 恭喜！整个流程已打通！ 🎉")
                print("="*30)
                print(f"✅ 捕获到的隐藏状态 (X) 的形状: {hidden_state_X.shape}")
                
                # 从嵌套列表中提取出真正的 DetokenizeOutput 对象
                skeleton_obj = predictions[0][0]
                print(f"✅ 成功提取骨架对象，类型为: {type(skeleton_obj)}")
                
                # 调用我们的新函数来保存 BVH 文件
                save_skeleton_to_bvh(skeleton_obj, output_bvh_path)
                
            else:
                print("❌ 错误: 推理完成但未能获取有效结果。")