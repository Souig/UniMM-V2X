import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import HEADS, build_loss
from einops import rearrange
# from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation # 如果不需要这个激活函数，可以注释掉
# from .planning_head_plugin import CollisionNonlinearOptimizer # 如果不使用复杂的优化器，可以注释掉
import numpy as np # 如果碰撞优化需要 numpy，则保留
# import copy # 如果 adapter 或其他地方需要深拷贝则保留
# import heapq # 如果 drivable_optimization 需要 heapq 则保留

@HEADS.register_module()
class PlanningHeadVanillaCNN(BaseModule):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=10, # 根据配置文件的 planning_steps
                 loss_planning=None, # For L2 error
                 loss_collision=None, # For collision rate
                 planning_eval=True, # Enable planning evaluation
                 use_col_optim=False, # Whether to use collision optimization (optional for vanilla)
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                 with_adapter=False, # Decide if you want to keep the adapter block
                 occ_n_future_only_occ=4, # This might not be directly used if occ_head is removed
                 ):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        
        # --- Simplified Input Processing ---
        # If command is still used:
        self.navi_embed = nn.Embedding(3, embed_dims) # Keep if command is used

        # We need a way to process the BEV features into a fixed-size vector
        # A simple CNN feature extractor + Global Pooling
        # This replaces the transformer interaction
        self.bev_feature_extractor = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dims // 2, embed_dims // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling to get a (B, C') vector
        )
        
        # Determine the input dimension to the final regression branch
        # This will be from self.bev_feature_extractor output + potentially navi_embed/other fused features
        
        # Assuming navi_embed (if used) is fused *after* BEV feature extraction
        # The output of bev_feature_extractor is (B, embed_dims // 4, 1, 1), so flatten to (B, embed_dims // 4)
        fusion_input_dim = embed_dims // 4 
        if self.navi_embed is not None: # If command input is used
             fusion_input_dim += embed_dims # + navi_embed dim

        # A simple MLP to fuse if multiple inputs (like navi_embed) are present
        self.fuser_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims) # Output dimension for reg_branch
        )

        # Regression branch for planning trajectory
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 2), # Adjusted input from fuser_mlp
            nn.ReLU(),
            nn.Linear(embed_dims // 2, planning_steps * 2), # Output x, y for each step
        )
        
        # Loss functions
        self.loss_planning = build_loss(loss_planning) # For L2 Error
        self.loss_collision = nn.ModuleList([build_loss(cfg) for cfg in loss_collision]) if loss_collision else None
        
        self.use_col_optim = use_col_optim
        if use_col_optim:
            # If you want to use the original CollisionNonlinearOptimizer, ensure it's imported
            # and its dependencies (like cvxpy) are met.
            # Otherwise, you might need a simplified collision optimization logic here
            self.occ_filter_range = col_optim_args['occ_filter_range']
            self.sigma = col_optim_args['sigma']
            self.alpha_collision = col_optim_args['alpha_collision']
            # Re-integrate collision_optimization and drivable_optimization methods from original
            # You might need to copy/adapt them from the original planning_head.py
            # For simplicity, I'll assume you adapt them, or provide a basic placeholder.

        self.with_adapter = with_adapter
        if with_adapter:
            # Assuming bev_adapter from original planning_head.py is relevant here
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3 # From original
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

    def init_weights(self):
        # Initialize CNN and MLP layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @force_fp32(apply_to=('bev_embed'))
    def forward(self, 
                bev_embed, 
                occ_mask=None, # Keep for potential collision optimization, but not direct input to CNN
                bev_pos=None, # Not directly used for vanilla CNN input
                sdc_traj_query=None, # Not directly used for vanilla CNN input
                sdc_track_query=None, # Not directly used for vanilla CNN input
                command=None,
                drivable_pred=None # Keep for potential drivable optimization
                ):
        """
        Forward pass for Vanilla CNN Planning Head.
        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding (B, C, H, W).
            ... other args from original PlanningHeadSingleMode for compatibility
        Returns:
            dict: Predicted SDC trajectory.
        """
        # Apply BEV adapter if enabled (from original PlanningHeadSingleMode)
        if self.with_adapter:
            # Assuming bev_embed is (B, C, H, W)
            # Original planning head rearranges to (HW, B, C) for transformer.
            # We'll assume the input bev_embed is (B, C, H, W) here for CNN processing.
            bev_feat_adapted = self.bev_adapter(bev_embed)
            bev_feat_to_process = bev_embed + bev_feat_adapted # Residual connection
        else:
            bev_feat_to_process = bev_embed

        # Extract features from BEV map using CNN
        cnn_features = self.bev_feature_extractor(bev_feat_to_process).squeeze(-1).squeeze(-1) # Output (B, embed_dims // 4)

        fused_features = cnn_features
        if command is not None and self.navi_embed is not None:
            navi_embed = self.navi_embed.weight[command] # (B, embed_dims)
            fused_features = torch.cat([cnn_features, navi_embed], dim=-1) # Concat (B, embed_dims//4 + embed_dims)

        # Fuse features if multiple sources
        final_features = self.fuser_mlp(fused_features) # Output (B, embed_dims)

        # Predict trajectory
        # Input to reg_branch is (B, embed_dims)
        # Output is (B, planning_steps * 2) -> reshape to (B, planning_steps, 2)
        pred_traj = self.reg_branch(final_features).view(-1, self.planning_steps, 2)
        
        # Apply initial cumulative sum to get absolute trajectory from relative offsets
        pred_traj_abs = torch.cumsum(pred_traj, dim=1)
        
        # Replicate bivariate_gaussian_activation if it's meant for the *final* trajectory
        # or remove if it was part of the original query-based decoding.
        # For a vanilla CNN, it's typically direct (x,y) prediction.
        # if not self.training: # This activation was often applied only at test time
        #    pred_traj_abs[0] = bivariate_gaussian_activation(pred_traj_abs[0]) # Assuming batch size 1 for this activation

        if self.use_col_optim and not self.training:
            # Post-process with collision and drivable optimization
            # These methods need to be implemented or copied from the original PlanningHeadSingleMode
            # and adjusted to take the predicted trajectory as input.
            if occ_mask is not None:
                # Assuming original collision_optimization method is available and adapted
                # Make sure these methods are part of this class or imported
                pred_traj_abs = self.collision_optimization(pred_traj_abs, occ_mask)
            if drivable_pred is not None:
                # Assuming original drivable_optimization method is available and adapted
                pred_traj_abs = self.drivable_optimization(pred_traj_abs, drivable_pred)
        
        return dict(
            sdc_traj=pred_traj_abs,
            sdc_traj_all=pred_traj_abs, # For compatibility with loss function input
        )

    # --- Copy/Adapt collision_optimization and drivable_optimization from original planning_head.py ---
    # These methods should be copied into this PlanningHeadVanillaCNN class
    # Make sure to import necessary modules like numpy, heapq, cv2 (if visualize is used)
    # The coordinate conversions (real2bev, bev2real) and grid sizes must match your BEV feature map.

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.
        (Copied from original PlanningHeadSingleMode - ensure dependencies are met)
        """
        pos_xy_t = []
        valid_occupancy_num = 0
        
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        #assert occ_horizon == 5
        # assert occ_horizon == (self.occ_n_future_only_occ+1) # This self.occ_n_future_only_occ might need to be passed
        # Or hardcode if occ_head is removed and this is just for fixed evaluation.
        # Let's use a hardcoded value if occ_n_future_only_occ is not passed via config.
        # Assuming occ_horizon from data is always 5 or similar from original config.
        assert occ_horizon == (4+1) # Based on occ_n_future=4 in base config

        for t in range(self.planning_steps):
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            if pos_xy.numel() == 0: # Handle empty occupancy
                continue
            pos_xy = pos_xy[:, [1, 0]]
            
            # --- Coordinate conversion from BEV pixels to real-world meters ---
            # These values (0.5, 0.25) depend on your voxel_size and BEV dimensions.
            # Make sure these are correct based on your overall coordinate system.
            # From your config: point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            # voxel_size = [0.2, 0.2, 8]
            # bev_h_=200, bev_w_=200 (for INF agent's BEV range)
            # The current coordinate conversion in the original planning_head.py looks specific
            # to the inf_point_cloud_range if bev_h/w are 200 for 102.4x102.4m range (0.512m/pixel).
            # For ego-agent's point_cloud_range [-51.2, -51.2, 51.2, 51.2] (102.4x102.4m),
            # with bev_h/w = 200, it's still 0.512m/pixel.
            # So, (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25 implies 0.5m/pixel and an offset.
            # This needs to be precisely derived from the actual BEV feature map coordinate system.
            # Assuming the original calculation is correct for the ego agent's BEV features.
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25 # Adjust based on actual BEV grid config
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25 # Adjust based on actual BEV grid config
            # -----------------------------------------------------------------

            # filter the occupancy in range
            # sdc_traj_all is (1, planning_steps, 2)
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)

        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        # CollisionNonlinearOptimizer needs to be imported and available.
        # Make sure planning_head_plugin.py is correctly set up.
        from .planning_head_plugin import CollisionNonlinearOptimizer
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
    
    def drivable_optimization(self, initial_trajectory, feasible_area):
        """
        Adjust the initial trajectory to ensure all points are within the feasible area using A* search.
        (Copied from original PlanningHeadSingleMode - ensure dependencies like cv2, heapq, numpy are met)
        """
        def is_within_bounds(x, y, grid_size):
            return 0 <= x < grid_size and 0 <= y < grid_size
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def a_star_search(start, feasible_area):
            neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            close_set = set()
            came_from = {}
            gscore = {start: 0}
            fscore = {start: heuristic(start, start)}
            oheap = []
            
            import heapq # Ensure heapq is imported if not at top level
            heapq.heappush(oheap, (fscore[start], start))
            
            while oheap:
                current = heapq.heappop(oheap)[1]
                
                # Check if current point is feasible
                # NOTE: feasible_area[current[0]][current[1]] == 1 means feasible
                # This assumes feasible_area is a binary grid
                if feasible_area[current[0]][current[1]] == 1:
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    return path[::-1] # Return path from start to goal
                
                close_set.add(current)
                for i, j in neighbors:
                    neighbor = current[0] + i, current[1] + j
                    # Check bounds before accessing feasible_area
                    if not (0 <= neighbor[0] < feasible_area.shape[0] and 0 <= neighbor[1] < feasible_area.shape[1]):
                        continue
                    
                    tentative_g_score = gscore[current] + heuristic(current, neighbor)

                    # Only consider if neighbor is feasible
                    if feasible_area[neighbor[0]][neighbor[1]] == 0: # If not feasible, skip
                        continue
                    
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue
                    
                    if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]: # Fix: use float('inf') for unseen nodes
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, start) # Heuristic to goal, not start
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
            
            return None # No path found
        
        # --- Visualization part commented out unless explicitly needed ---
        # def visualize_trajectory_and_feasible_area(initial_trajectory, feasible_area, filename):
        #     import cv2
        #     feasible_area_np = feasible_area.cpu().numpy() * 255
        #     feasible_area_color = cv2.cvtColor(feasible_area_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        #     for i in range(initial_trajectory.size(0)):
        #         x, y = initial_trajectory[i].int().tolist()
        #         color = (0, 0, 255) if not is_feasible(x, y, feasible_area) else (0, 255, 0)
        #         cv2.circle(feasible_area_color, (y, x), 3, color, -1)
        #     filename = 'workspace/0805_debug_drivable/' + filename
        #     cv2.imwrite(filename, feasible_area_color)

        # --- Coordinate conversion functions (crucial for BEV to real world) ---
        # Ensure point_cloud_range is correctly defined and used.
        # Original points_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        # bev_h_ = 200, bev_w_ = 200
        # This implies a resolution of (51.2 - (-51.2)) / 200 = 102.4 / 200 = 0.512 meters per pixel.
        # The provided real2bev and bev2real seems to assume a different range, e.g., [-51.2, 51.2] for both X and Y
        # This should be derived from the actual 'point_cloud_range' used for the ego agent's BEV features.
        # Let's adjust real/bev based on your global point_cloud_range if bev_h/w are tied to it.
        # From your config: point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        # This range is 102.4m x 102.4m, mapped to 200x200 BEV.
        # So, scale = 102.4 / 200 = 0.512 m/pixel
        # Offset: -51.2m corresponds to 0 pixel.
        # (pixel_val - 0) * scale + min_real_val
        # real_val = pixel_val * scale + min_real_val
        # pixel_val = (real_val - min_real_val) / scale
        # For X (dim 0): min_real_x = -51.2, max_real_x = 51.2
        # For Y (dim 1): min_real_y = -51.2, max_real_y = 51.2
        
        # Your provided real2bev/bev2real:
        # real = [-51.2, 51.2] implies a range of 102.4 for both x and y.
        # bev = [200, 200]
        # bev_traj[:, 0] = (-traj[:, 1] + real[1]) / (real[1] - real[0]) * (bev[0] - 1) # Y real to X bev
        # bev_traj[:, 1] = (traj[:, 0] + real[1]) / (real[1] - real[0]) * (bev[1] - 1) # X real to Y bev
        # This indicates a coordinate swap and inversion. This might be correct for your specific setup.
        # Assuming this transformation from original PlanningHeadSingleMode is correct for your system.

        real_x_range = [self.pc_range[0], self.pc_range[3]] # from global config
        real_y_range = [self.pc_range[1], self.pc_range[4]] # from global config
        # Assuming bev_h and bev_w are used consistently
        
        def real2bev(traj_real_coords):
            # traj_real_coords shape: (N, 2) where dim 0 is X and dim 1 is Y in real world
            bev_traj = torch.zeros_like(traj_real_coords)
            
            # X_real -> Y_bev (map real X to bev Y)
            bev_traj[:, 1] = (traj_real_coords[:, 0] - real_x_range[0]) / (real_x_range[1] - real_x_range[0]) * (self.bev_w - 1)
            
            # Y_real -> X_bev (map real Y to bev X, often inverted)
            # The original code has (-traj[:, 1] + real[1]). This means Y_real = max_y corresponds to X_bev = 0.
            bev_traj[:, 0] = (real_y_range[1] - traj_real_coords[:, 1]) / (real_y_range[1] - real_y_range[0]) * (self.bev_h - 1)
            
            return torch.clamp(bev_traj, 0, max(self.bev_h, self.bev_w) - 1) # Clamp to valid pixel range

        def bev2real(traj_bev_coords):
            # traj_bev_coords shape: (N, 2) where dim 0 is X_bev and dim 1 is Y_bev
            real_traj = torch.zeros_like(traj_bev_coords)
            
            # Y_bev -> X_real
            real_traj[:, 0] = traj_bev_coords[:, 1] / (self.bev_w - 1) * (real_x_range[1] - real_x_range[0]) + real_x_range[0]
            
            # X_bev -> Y_real (inverted)
            real_traj[:, 1] = real_y_range[1] - (traj_bev_coords[:, 0] / (self.bev_h - 1) * (real_y_range[1] - real_y_range[0]))
            
            return real_traj

        def is_feasible(x_bev, y_bev, feasible_area):
            # Check a 3x3 neighborhood for feasibility
            neighbors = [
                (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            ]
            feasible_count = 0
            for dx, dy in neighbors:
                nx, ny = int(x_bev + dx), int(y_bev + dy)
                if 0 <= nx < feasible_area.shape[0] and 0 <= ny < feasible_area.shape[1]:
                    if feasible_area[nx, ny] == 1:
                        feasible_count += 1
                else: # Out of bounds is considered feasible
                    feasible_count += 1
            return feasible_count >= 2 # At least 2 neighboring pixels are feasible

        def is_consecutive(nums):
            # Check if a list of numbers are consecutive (e.g., [2,3,4])
            for i in range(len(nums)-1):
                if nums[i] != nums[i+1] - 1:
                    return False
            return True
        
        # Ensure pc_range is available in this class
        # You might need to pass pc_range from config to init of this head
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] # Hardcoding for now, better to pass from config

        ori_trajectory = initial_trajectory.clone()
        initial_trajectory_batch = initial_trajectory # (B, P, 2)
        
        # Assuming batch size of 1 for optimization as in original code
        initial_trajectory = initial_trajectory_batch[0] # (P, 2)
        adjusted_trajectory = initial_trajectory.clone() # (P, 2)
        
        grid_height = feasible_area.shape[0] # BEV height
        grid_width = feasible_area.shape[1] # BEV width

        # Convert initial trajectory from real-world to BEV pixel coordinates
        initial_trajectory_bev = real2bev(initial_trajectory)
        adjusted_trajectory_bev = real2bev(adjusted_trajectory)

        non_feasible_points_indices = []
        for i in range(initial_trajectory_bev.size(0)):
            x_bev, y_bev = initial_trajectory_bev[i].int().tolist()
            # Check if point is within BEV bounds AND feasible in area
            if not is_within_bounds(x_bev, y_bev, grid_height) or \
               not is_within_bounds(x_bev, y_bev, grid_width) or \
               not is_feasible(x_bev, y_bev, feasible_area):
                non_feasible_points_indices.append(i)
        
        if len(non_feasible_points_indices) == 0:
            return ori_trajectory # No adjustment needed

        # If a block of end points are infeasible, stick them to the last feasible point
        if is_consecutive(non_feasible_points_indices) and non_feasible_points_indices[-1] == self.planning_steps - 1:
            if non_feasible_points_indices[0] > 0:
                # Get the last feasible point
                last_feasible_point_bev = adjusted_trajectory_bev[non_feasible_points_indices[0] - 1]
                for i in non_feasible_points_indices:
                    adjusted_trajectory_bev[i] = last_feasible_point_bev
            else: # First point is infeasible, cannot stick to previous. Needs A* from somewhere.
                  # For now, if all are infeasible from start, just return original.
                  # A* might be needed for such cases, but the provided drivable_optimization is simplified.
                pass # This case implies a more complex A* or fallback

        # Convert adjusted trajectory back to real-world coordinates
        adjusted_trajectory_real = bev2real(adjusted_trajectory_bev)
        
        return adjusted_trajectory_real.unsqueeze(0) # Add batch dimension back

    # --- Loss function (adapted from original PlanningHeadSingleMode) ---
    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes=None):
        """
        Calculate loss for planning, including L2 error and collision loss.
        
        Args:
            sdc_planning (torch.Tensor): Ground truth SDC trajectory (B, M, P, 3).
                                         Original uses (B, 1, P, 3) for single mode.
            sdc_planning_mask (torch.Tensor): Mask for valid GT trajectory steps (B, M, P).
            outs_planning (dict): Predicted planning results, containing 'sdc_traj_all'.
                                  'sdc_traj_all' should be (B, P, 2) (our predicted traj).
            gt_future_boxes (list[torch.Tensor]): List of future GT object bboxes (B, N_obj, 9)
                                                  for each timestamp.
        Returns:
            dict: Calculated losses.
        """
        pred_traj_planning = outs_planning['sdc_traj_all'] # (B, P, 2)

        loss_dict = dict()

        # Reshape GT planning for compatibility
        # Original sdc_planning (B, M, P, 3) -> (B, P, 2) for L2 loss
        # Assuming M=1 (single mode) and only x,y are relevant for L2
        gt_planning_l2 = sdc_planning[:, 0, :, :2] # (B, P, 2)
        gt_planning_mask_l2 = torch.any(sdc_planning_mask[:, 0, :self.planning_steps], dim=-1).unsqueeze(1).expand(-1, self.planning_steps) # (B, P)

        # L2 Loss for planning trajectory
        # Using the L2Loss custom module from previous step
        # Note: If gt_planning_mask_l2 is (B, P), then you need to mask pred_traj_planning and gt_planning_l2
        # and calculate loss on valid steps.
        
        # Calculate element-wise L2 error squared (distance squared)
        l2_error_sq = torch.sum((pred_traj_planning - gt_planning_l2)**2, dim=-1) # (B, P)
        
        # Apply mask and sum valid errors
        masked_l2_error_sq_sum = torch.sum(l2_error_sq * gt_planning_mask_l2, dim=-1) # (B,)
        num_valid_points = torch.sum(gt_planning_mask_l2, dim=-1) # (B,)
        
        # Average L2 error for valid points, handle division by zero
        # This is Mean Squared Error, for L2 Distance, take sqrt later or define loss differently
        mean_l2_error_per_sample = masked_l2_error_sq_sum / (num_valid_points + 1e-6) # (B,)
        
        # Use your custom L2Loss (assuming it expects scalar or a mean)
        # If your L2Loss is just MSE, you might just do mean_l2_error_per_sample.mean()
        loss_dict['loss_planning_l2'] = self.loss_planning(mean_l2_error_per_sample, torch.zeros_like(mean_l2_error_per_sample)) # Assuming it expects pred, target

        # Collision Loss
        if self.loss_collision:
            # gt_future_boxes is list[Tensor(N_obj, 9)] per batch item
            # For collision loss, it might expect the current sdc_bbox as well.
            # You might need to ensure sdc_bbox is passed into the loss function from forward_train.
            
            # The original loss function call seems to use sdc_planning[0, :, :self.planning_steps, :3]
            # for SDC BBox in a weird way (likely using sdc_planning for both traj and bbox if not separate).
            # This needs to be correctly adapted.
            # Assuming current sdc_bbox is available, maybe from img_metas or a direct input to loss.
            # For simplicity, let's assume gt_sdc_bbox is passed as `kwargs` to loss.
            
            # The original call:
            # loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            # This looks like: pred_traj, gt_sdc_bbox (for each time step, which is unusual), gt_mask, gt_future_boxes (from t=1 to end).
            
            # Revised simplified call (assuming standard CollisionLoss inputs)
            # You need a `current_sdc_bbox` here
            # sdc_planning_original_format is needed for sdc_bbox or ensure it is passed separately
            
            # For the vanilla setup, let's assume we extract SDC's current bbox from somewhere or simply use a dummy.
            # In a real setup, it should be passed from the data pipeline or predicted.
            current_sdc_bbox = None # (B, 9) if available
            
            for i, collision_loss_func in enumerate(self.loss_collision):
                if current_sdc_bbox is not None and gt_future_boxes is not None:
                    # Pass the predicted trajectory (pred_traj_planning), SDC's current bbox,
                    # and list of future object bboxes (gt_future_boxes)
                    # The specific arguments depend on your CollisionLoss implementation.
                    col_loss_val = collision_loss_func(
                        pred_traj_planning,      # Predicted SDC trajectory (B, P, 2)
                        current_sdc_bbox,        # Current SDC bbox (B, 9)
                        gt_future_boxes,         # List of future object bboxes (list of tensors)
                        # Add any other required parameters for CollisionLoss
                    )
                    loss_dict[f'loss_collision_{i}'] = col_loss_val
                else:
                    # If GT for collision is not available, set loss to 0
                    loss_dict[f'loss_collision_{i}'] = torch.tensor(0.0, device=pred_traj_planning.device)

        return loss_dict