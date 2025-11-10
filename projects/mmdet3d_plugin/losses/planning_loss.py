import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2', weight=1.0):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
        self.weight = weight
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        # with torch.no_grad():
        #     for t in range(sdc_traj.shape[1]):
        #         err_t = torch.sqrt(torch.sum((sdc_traj[:, t, :2] - gt_sdc_fut_traj[:, t, :2])**2, dim=-1))
        #         print(f"L2 Error at t={t}: {err_t.mean().item():.3f}")
        weights = torch.tensor([1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0]).to(sdc_traj.device)
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        return self.weight * torch.sum(weights * err * mask)/(torch.sum(mask) + 1e-5)
    
    
@LOSSES.register_module()
class PlanningLossNew(nn.Module):
    def __init__(self,
                 loss_type='L2',
                 l2_weight=1.0,
                 temp_weight=1.0,
                 phys_weight=0.5):
        super(PlanningLossNew, self).__init__()
        self.loss_type = loss_type
        self.temp_weight = temp_weight
        self.phys_weight = phys_weight
        self.l2_weight = l2_weight

    def forward(self, sdc_traj, gt_sdc_fut_traj, mask, future_obs_t1=None):
        """
        sdc_traj: (B, T, 2 or 3), predicted trajectory
        gt_sdc_fut_traj: (B, T, 2 or 3), ground truth future trajectory
        mask: (B, T), validity mask
        future_obs_t1: (B, 2), optional, used for temporal consistency
        """

        # -------- L2 trajectory loss -------- #
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        l2_dist = torch.sqrt(torch.sum(err ** 2, dim=-1))  # (B, T)
        loss_l2 = torch.sum(l2_dist * mask) / (torch.sum(mask) + 1e-5)

        # -------- Temporal Consistency Loss -------- #
        future_obs_t1 = gt_sdc_fut_traj[:, 1, :2]
        if future_obs_t1 is not None:
            pred_pos_0 = sdc_traj[:, 0, :2]  # (B, 2)
            temp_diff = pred_pos_0 - future_obs_t1  # (B, 2)
            loss_temp = (temp_diff ** 2).sum(dim=-1).mean()
        else:
            loss_temp = 0.0

        # -------- Physics Smoothness Loss (Jerk) -------- #
        vel = sdc_traj[:, 1:, :2] - sdc_traj[:, :-1, :2]        # (B, T-1, 2)
        acc = vel[:, 1:, :] - vel[:, :-1, :]                    # (B, T-2, 2)
        jerk = acc[:, 1:, :] - acc[:, :-1, :]                   # (B, T-3, 2)
        loss_phys = (jerk ** 2).sum(dim=-1).mean()

        # -------- Combine Loss -------- #
        total_loss = self.l2_weight * loss_l2 + self.temp_weight * loss_temp + self.phys_weight * loss_phys
        return total_loss


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        dump_sdc = []
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_bbox(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        
        xi1, yi1 = min(xa1, xb1), min(ya1, yb1)
        xi2, yi2 = max(xa2, xb2), max(ya2, yb2)
        intersect = max((xi1 - xi2), xi1.new_zeros(1, ).to(xi1.device)) * max((yi1 - yi2), xi1.new_zeros(1,).to(xi1.device))
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        # corners = torch.tensor([
        #     [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]
        # ]).to(x.device) # 4,2
        corners = torch.tensor([
            [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2,-w/2]
        ]).to(x.device) # 4,2
        rot_mat = torch.tensor(
            [[torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]]
        ).to(x.device)
        new_corners = rot_mat @ corners.T + torch.tensor(bbox[:2])[:, None].to(x.device)
        return new_corners.T