export PYTHONPATH=$(pwd)
# export CUDA_VISIBLE_DEVICES=0
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# 运行 Python 推理脚本

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29502 \
    tools/inference.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_old_mode_inference_wo_label.py \
    projects/work_dirs_e2e_univ2x/univ2x_coop_e2e_new_motion/epoch_3_3.06+4.10.pth \
    --out output/results_track2_epoch3_3.06+4.10.pkl \
    --launcher pytorch