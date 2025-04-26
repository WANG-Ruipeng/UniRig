from collections import defaultdict

import torch.distributed
import lightning as L
import os
import torch
import numpy as np
from torch import Tensor, FloatTensor, LongTensor
from typing import Dict, Union, List, Literal
from lightning.pytorch.callbacks import BasePredictionWriter

from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from ..data.order import OrderConfig, get_order
from ..data.raw_data import RawSkin, RawData
from ..data.exporter import Exporter
from ..model.spec import ModelSpec

class SkinSystem(L.LightningModule):
    
    def __init__(
        self,
        steps_per_epoch: int,
        model: ModelSpec,
        output_path: Union[str, None]=None,
        record_res: Union[bool]=False,
        val_interval: Union[int, None]=None,
        val_start_from: Union[int, None]=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.steps_per_epoch    = steps_per_epoch
        self.model              = model
        self.output_path        = output_path
        self.record_res         = record_res
        self.val_interval       = val_interval
        self.val_start_from     = val_start_from
        
        if self.record_res:
            assert self.output_path is not None, "record_res is True, but output_path in skin is None"
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        res = self.model.predict_step(batch)
        
        if isinstance(res, list):
            return {
                'skin_pred': res,
            }
        elif isinstance(res, dict):
            assert 'skin_pred' in res, f"expect key 'skin_pred' in prediction from {self.model.__class__}, found: {res.keys()}"
            return res
        else:
            assert 0, f"expect type of prediction from {self.model.__class__} to be a list or dict, found: {type(res)}"

class SkinWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Union[str, None],
        save_name: str,
        order_config: Union[OrderConfig, None]=None,
        **kwargs
    ):
        super().__init__('batch')
        self.output_dir         = output_dir
        self.npz_dir            = kwargs.get('npz_dir', None)
        self.user_mode          = kwargs.get('user_mode', False)
        self.output_name        = kwargs.get('output_name', None) # for a single name
        self.save_name          = save_name
        self.add_num            = kwargs.get('add_num', False)
        self.export_npz         = kwargs.get('export_npz', True)
        self.export_fbx         = kwargs.get('export_fbx', False)
        if order_config is not None:
            self.order = get_order(config=order_config)
        else:
            self.order = None
        
        self._epoch = 0

    def write_on_batch_end(self, trainer, pl_module: SkinSystem, prediction: List[Dict], batch_indices, batch, batch_idx, dataloader_idx):
        assert 'path' in batch
        paths: List[str] = batch['path']
        data_names: List[str] = batch['data_name']
        joints: FloatTensor = batch['joints']
        num_bones: LongTensor = batch['num_bones']
        num_faces: LongTensor = batch['num_faces']
        num_points: LongTensor = batch['num_points']
        tails: FloatTensor = batch['tails']
        parents_list: LongTensor = batch['parents'] # -1 represents root
        vertices: FloatTensor = batch['origin_vertices']
        sampled_vertices: FloatTensor = batch['vertices']
        faces: LongTensor = batch['origin_faces']
        
        joints = joints.detach().cpu().numpy()
        tails = tails.detach().cpu().numpy()
        parents_list = parents_list.detach().cpu().numpy()
        num_bones = num_bones.detach().cpu().numpy()
        num_faces = num_faces.detach().cpu().numpy()
        vertices = vertices.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()

        skin_pred_list: List = prediction['skin_pred']
        ret_sampled_vertices = prediction.get('sampled_vertices', None)
        if ret_sampled_vertices is not None:
            assert isinstance(ret_sampled_vertices, Tensor)
            sampled_vertices = ret_sampled_vertices
        if isinstance(sampled_vertices, Tensor):
            sampled_vertices = sampled_vertices.type(torch.float32).detach().cpu().numpy()
        for (id, skin_pred) in enumerate(skin_pred_list):
            if isinstance(skin_pred, Tensor):
                skin_pred = skin_pred.type(torch.float32).detach().cpu().numpy()
                
            # TODO: add custom post-processing here
            
            # resample
            N = num_points[id]
            J = num_bones[id]
            F = num_faces[id]
            o_vertices = vertices[id, :N]

            _parents = parents_list[id]
            parents = []
            for i in range(J):
                if _parents[i] == -1:
                    parents.append(None)
                else:
                    parents.append(_parents[i])

            skin_resampled = reskin(
                sampled_vertices=sampled_vertices[id],
                vertices=o_vertices,
                parents=parents,
                faces=faces[id, :F],
                sampled_skin=skin_pred,
                sample_method='median',
                alpha=2.0,
                threshold=0.03,
            )
            
            def make_path(save_name: str, suffix: str, trim: bool=False):
                if trim:
                    path = os.path.relpath(paths[id], self.npz_dir)
                else:
                    path = paths[id]

                if self.output_dir is not None:
                    path = os.path.join(self.output_dir, path)
                
                if self.add_num:
                    path = os.path.join(path, f"{save_name}_{self._epoch}.{suffix}")
                else:
                    path = os.path.join(path, f"{save_name}.{suffix}")
                return path
            
            raw_data = RawSkin(skin=skin_pred, vertices=sampled_vertices[id], joints=joints[id, :J])
            if self.export_npz is not None:
                raw_data.save(path=make_path(self.export_npz, 'npz'))
            if self.export_fbx is not None:
                try:
                    exporter = Exporter()
                    names = RawData.load(path=os.path.join(paths[id], data_names[id])).names
                    if names is None:
                        names = [f"bone_{i}" for i in range(J)]
                    if self.user_mode:
                        if self.output_name is not None:
                            path = self.output_name
                        else:
                            path = make_path(self.save_name, 'fbx', trim=True)
                    else:
                        path = make_path(self.export_fbx, 'fbx')
                    exporter._export_fbx(
                        path=path,
                        vertices=o_vertices,
                        joints=joints[id, :J],
                        skin=skin_resampled,
                        parents=parents,
                        names=names,
                        faces=faces[id, :F],
                        group_per_vertex=4,
                        tails=tails[id, :J],
                        use_extrude_bone=False,
                        use_connect_unique_child=False,
                        # do_not_normalize=True,
                    )
                except Exception as e:
                    print(str(e))
    
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        self._epoch += 1

def reskin(
    sampled_vertices: ndarray,
    vertices: ndarray,
    parents: List[Union[None, int]],
    faces: ndarray,
    sampled_skin: ndarray,
    sample_method: Literal['mean', 'median']='mean',
    **kwargs,
) -> ndarray:
    nearest_samples = kwargs.get('nearest_samples', 7)
    iter_steps = kwargs.get('iter_steps', 1)
    threshold = kwargs.get('threshold', 0.01)
    alpha = kwargs.get('alpha', 2)
    
    assert sample_method in ['mean', 'median']
    
    N = vertices.shape[0]
    J = sampled_skin.shape[1]
    if sample_method == 'mean':
        tree = cKDTree(sampled_vertices)
        dis, nearest = tree.query(vertices, k=nearest_samples, p=2)
        # weighted sum
        weights = np.exp(-alpha * dis)  # (N, nearest_samples)
        weight_sum = weights.sum(axis=1, keepdims=True)
        sampled_skin_nearest = sampled_skin[nearest]
        skin = (sampled_skin_nearest * weights[..., np.newaxis]).sum(axis=1) / weight_sum
    elif sample_method == 'median':
        tree = cKDTree(sampled_vertices)
        dis, nearest = tree.query(vertices, k=nearest_samples, p=2)
        skin = np.median(sampled_skin[nearest], axis=1)
    else:
        assert 0
    
    # (from, to)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0) # (2*F*3, 2)

    # diffusion in neighbours
    for _ in range(iter_steps):
        sum_skin = skin.copy()
        for i in reversed(range(J)):
            p = parents[i]
            if p is None:
                continue
            sum_skin[:, p] += sum_skin[:, i]
        # (2*F*3, J)
        # only transfer from hotter to cooler
        mask = sum_skin[edges[:, 1]] < sum_skin[edges[:, 0]]
        neighbor_skin = np.zeros_like(sum_skin)  # (N, J)
        neighbor_co = np.zeros((N, J), dtype=np.float32)

        dis = np.sqrt(((vertices[edges[:, 1]] - vertices[edges[:, 0]])**2).sum(axis=1, keepdims=True))
        co = np.exp(-dis * alpha)

        neighbor_skin[edges[:, 1]] += sum_skin[edges[:, 0]] * co * mask
        neighbor_co[edges[:, 1]] += co * mask

        sum_skin = (sum_skin + neighbor_skin) / (1. + neighbor_co)
        for i in range(J):
            p = parents[i]
            if p is None:
                continue
            sum_skin[:, p] -= sum_skin[:, i]
            skin = sum_skin / sum_skin.sum(axis=-1, keepdims=True)

    # avoid 0-skin
    mask = (skin>=threshold).any(axis=-1, keepdims=True)
    skin[(skin<threshold)&mask] = 0.
    skin = skin / skin.sum(axis=-1, keepdims=True)
    
    return skin