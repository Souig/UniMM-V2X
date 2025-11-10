# Assuming you define Expert and MoE classes similar to the previous explanation
# You might want to place these in a separate file like 'moe_layer.py'
# and then import them here: `from .moe_layer import Expert, MoE`

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, build_feedforward_network, build_attention
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn import build_norm_layer

logger = logging.getLogger(__name__)

def build_activation_network(act_cfg):
    """Build activation network."""
    from mmcv.cnn.bricks.activation import build_activation_layer
    if isinstance(act_cfg, dict):
        return build_activation_layer(act_cfg)
    elif isinstance(act_cfg, nn.Module):
        return act_cfg
    else:
        raise TypeError(f'Unsupported activation type: {type(act_cfg)}. Expected dict or nn.Module.')

class Expert(nn.Module):

    def __init__(self, d_model, d_ff, act_cfg=dict(type='ReLU', inplace=True), ffn_dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.act = build_activation_network(act_cfg)
        self.dropout = nn.Dropout(ffn_dropout) if ffn_dropout > 0 else None
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU() 

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x)))) if self.dropout else self.w2(self.act(self.w1(x)))

class MoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k, d_ff=2048, 
                 act_cfg=dict(type='ReLU', inplace=True), ffn_dropout=0.0,
                 load_balance_loss_weight=0.0, batch_first=False):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k 
        self.load_balance_loss_weight = load_balance_loss_weight
        
        self.gate = nn.Linear(d_model, num_experts)
        
        self.experts = nn.ModuleList([Expert(d_model, d_ff, act_cfg, ffn_dropout) for _ in range(num_experts)])

    def forward(self, x, residual=None):
        # x : (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model) if batch_first is True

        # if not self.batch_first:
        #    x = x.transpose(0, 1) # (batch_size, seq_len, d_model)

        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model) # (num_tokens, d_model)
        num_tokens = x_flat.shape[0]  # total number of tokens in the batch

        # 1. calculate gate logits for each expert
        gate_logits = self.gate(x_flat) # (num_tokens, num_experts)
        gumbel_noise = -torch.empty_like(gate_logits).exponential_().log()
        noisy_logits = gate_logits + gumbel_noise
        
        # 2. choose top-k experts for each token
        # topk_weights: (num_tokens, top_k), topk_indices: (num_tokens, top_k)
        topk_weights, topk_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        
        # 3. softmax the weights
        topk_weights = F.softmax(topk_weights, dim=-1) # (num_tokens, top_k)

        # 4. prepare output tensor
        # topk_indices: (num_tokens, top_k) -> (num_tokens * top_k)
        output_flat = torch.zeros_like(x_flat) # (num_tokens, d_model)
        flat_topk_indices = topk_indices.view(-1)
        
        # repeat the input for each token to match the top_k experts
        # x_flat (num_tokens, d_model) -> (num_tokens * top_k, d_model)
        repeated_x_flat = x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.d_model)
        
        # create index for each token's expert output, indicating where it should be added back to the output
        # (num_tokens, top_k) -> (num_tokens * top_k)
        token_indices = torch.arange(x_flat.shape[0], device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)
        
        expert_counts = torch.zeros(self.num_experts, dtype=torch.int64, device=x.device)
        expert_counts.scatter_add_(0, topk_indices.view(-1), torch.ones_like(topk_indices.view(-1), dtype=torch.int64))
        active_experts = (expert_counts > 0).sum().item()
        avg_expert_usage = expert_counts.float().mean().item() if x_flat.shape[0] > 0 else 0
        # randomly print
        i = torch.randint(0, 5000, (1,)).item()  # Randomly select an iteration number for logging
        if i  == 250:
            print(f'Active Experts: {active_experts}/{self.num_experts}, Average Expert Usage: {avg_expert_usage:.2f} (time/expert), Expert Counts: {expert_counts.tolist()}' )
        
        all_expert_probs = F.softmax(gate_logits, dim=-1)  # (num_tokens, num_experts)
        one_hot_topk_selection = torch.zeros(
            num_tokens, self.num_experts, device=x.device, dtype=x.dtype
        ).scatter_(1, topk_indices, 1)
        expert_importance = all_expert_probs.sum(dim=0)  # (num_experts,)
        expert_load = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        expert_load.scatter_add_(0, flat_topk_indices, topk_weights.view(-1))
        mean_expert_prob_per_token = all_expert_probs.mean(dim=0)
        # if num_tokens > 0:
        #     expert_assigned_fraction = expert_load / num_tokens
        #     expert_importance_sum = all_expert_probs.sum(dim=0)
        #     expert_load_sum = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        #     expert_load_sum.scatter_add_(0, flat_topk_indices, topk_weights.view(-1))
        #     aux_loss = torch.sum(expert_importance_sum * expert_load_sum) - \
        #         torch.sum(expert_importance_sum) * torch.sum(expert_load_sum)
        #     self.moe_load_balance_loss = self.load_balance_loss_weight * aux_loss
        # else:
        #     self.moe_load_balance_loss = torch.tensor(0.0, device=x.device)
        
        importance = all_expert_probs.sum(dim=0) / num_tokens  # shape: (num_experts,)
        load = expert_load / num_tokens  # shape: (num_experts,)

        importance_mean = importance.mean()
        load_mean = load.mean()

        aux_loss = ((importance - importance_mean) ** 2).mean() + ((load - load_mean) ** 2).mean()
        self.moe_load_balance_loss = self.load_balance_loss_weight * aux_loss

        for i, expert in enumerate(self.experts):
            expert_assigned_mask = (flat_topk_indices == i)
            
            expert_inputs = repeated_x_flat[expert_assigned_mask]
            expert_output_indices = token_indices[expert_assigned_mask]
            
            if expert_inputs.numel() > 0: 
                current_expert_output = expert(expert_inputs) # (num_assigned_tokens, d_model)
                
                expert_weights_for_tokens = topk_weights.view(-1)[expert_assigned_mask]
                
                weighted_expert_output = current_expert_output * expert_weights_for_tokens.unsqueeze(-1)
                
                # output_flat[expert_output_indices] += weighted_expert_output
                output_flat.scatter_add_(0, expert_output_indices.unsqueeze(-1).expand(-1, self.d_model), weighted_expert_output)
        
        # restore the original shape
        output = output_flat.view(original_shape)

        # if residual is not None:
        #     output = output + residual

        return output