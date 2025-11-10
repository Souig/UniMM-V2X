import torch
import torch.nn as nn
import numpy as np
import pickle
from projects.mmdet3d_plugin.models.utils.functional import (
    bivariate_gaussian_activation,
    norm_points,
    pos2posemb2d,
    anchor_coordinate_transform
)
class TrajFusion(nn.Module):
    def __init__(self, embed_dims=256, anchor_info_path=''):
        super(TrajFusion, self).__init__()
        self.embed_dims = embed_dims
        
        anchor_infos = pickle.load(open(anchor_info_path, 'rb'))
        self.kmeans_anchors = torch.stack(
            [torch.from_numpy(a) for a in anchor_infos["anchors_all"]])  # Nc, Pc, steps, 2

        self.scene_level_offset_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        
        self.query_fusion_attn = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dims,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.query_fusion_cross_attn = torch.nn.MultiheadAttention(embed_dim=self.embed_dims, num_heads=8, batch_first=True)
        self.proj_pos = torch.nn.Linear(self.embed_dims, self.embed_dims)
    
    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        """
        Group mode query positions based on the input bounding box results.
        
        Args:
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
            mode_query_pos (torch.Tensor): A tensor of shape (B, A, G, P, D) representing the mode query positions.
        
        Returns:
            torch.Tensor: A tensor of shape (B, A, P, D) representing the classified mode query positions.
        """
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)
        # TODO: vectorize this
        # group the embeddings based on the class
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            grouped_label = self.cls2group[label]
            grouped_mode_query_pos = []
            for j in range(agent_num):
                grouped_mode_query_pos.append(
                    mode_query_pos[i, j, grouped_label[j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)
    
    def forward(self, other_agent_results, outs_motion, track_query):
        for other_agent_name, other_agent_result in other_agent_results.items():
            ego2other_rt = other_agent_result['ego2other_rt']
            other_agent_pc_range = other_agent_result['pc_range']
            other_agent_motion = other_agent_result.get('unimmv2x_motion', None)
            other_agent_track = other_agent_result.get('unimmv2x_track', None)
            if other_agent_motion.get('traj_query', None) is not None:
                traj_query_self = outs_motion['traj_query'][-1]   # [B=1, A, M, D]
                traj_query_other = other_agent_motion['traj_query'][-1]  # [1, A', M, D]
                
                agent_level_anchors = self.kmeans_anchors.to(track_query.dtype).to(track_query.device)  # [G, P, T, 2]
                agent_level_anchors = agent_level_anchors.view(self.num_anchor_group, self.num_anchor, self.predict_steps, 2)
                
                other_track_boxes = other_agent_track['track_bbox_results']
                anchor_other = anchor_coordinate_transform(
                    agent_level_anchors, other_track_boxes, with_translation_transform=False)  # [1, A', G, P, T, 2]

                R = torch.tensor(np.linalg.inv(ego2other_rt[0].cpu().numpy().T), dtype=traj_query_self.dtype, device=traj_query_self.device)  # [2, 2]
                R2x2 = R[:2, :2].unsqueeze(0)  # [1, 2, 2]

                anchor_other_xy = anchor_other[..., -1, :]  # [1, A', G, P, 2]
                anchor_other_xy = anchor_other_xy.permute(0, 1, 2, 3, 4)  # [1, A', G, P, 2]
                anchor_other_xy_rot = torch.matmul(anchor_other_xy.unsqueeze(-2), R2x2.unsqueeze(1).unsqueeze(1))  # [1, A', G, P, 1, 2] @ [1, 2, 2]
                anchor_other_xy_rot = anchor_other_xy_rot.squeeze(-2)  # [1, A', G, P, 2]

                anchor_other_xy_rot_norm = norm_points(anchor_other_xy_rot, other_agent_pc_range)
                traj_query_pos_other = self.scene_level_offset_embedding_layer(
                    pos2posemb2d(anchor_other_xy_rot_norm)
                )  # [1, A', G, P, D]

                traj_query_pos_other = self.group_mode_query_pos(
                    other_track_boxes, traj_query_pos_other)  # [1, A', P, D]
                
                traj_q_self_flat = traj_query_self.view(1, -1, 256)         # [1, A*M, D]
                traj_q_other_flat = torch.cat([traj_query_other, traj_query_pos_other], dim=1).view(1, -1, self.embed_dims)  # [1, A'*M, D]
                traj_q_other_flat = self.proj_pos(traj_q_other_flat)  # [1, A'*M, D]
                    
                fused = self.query_fusion_attn(torch.cat([traj_q_self_flat, traj_q_other_flat], dim=1))[:, :traj_q_self_flat.shape[1]]
                fused, _ = self.query_fusion_cross_attn(query=fused, key=track_query[:, -1], value=track_query[:, -1])  # [1, A*M, D]
                fused = fused.view_as(traj_query_self)  # [1, A, M, D]
                
                outs_motion['traj_query'] = outs_motion['traj_query'].clone()
                outs_motion['traj_query'][-1] = fused
        return outs_motion
            