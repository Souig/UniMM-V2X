# UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative
Autonomous driving holds transformative potential but remains fundamentally constrained by the limited perception and isolated decision-making with standalone intelligence. While recent multi-agent approaches introduce cooperation, they often focus merely on perception-level tasks, overlooking the alignment with downstream planning and control, or fall short in leveraging the full capacity of the recent emerging end-to-end autonomous driving. In this paper, we present UniMM-V2X, a novel end-to-end multi-agent framework that enables hierarchical cooperation across perception, prediction, and planning. At the core of our framework is a multi-level fusion strategy that unifies perception and prediction cooperation, allowing agents to share queries and reason cooperatively for consistent and safe decision-making. To adapt to diverse downstream tasks and further enhance the quality of multi-level fusion, we incorporate a Mixture-of-Experts (MoE) architecture to dynamically enhance the BEV representations. We further extend MoE into the decoder to better capture diverse motion patterns. Extensive experiments on the DAIR-V2X dataset demonstrate our approach achieves state-of-the-art (SOTA) performance with a 39.7\% improvement in perception accuracy, a 7.2\% reduction in prediction error, and a 33.2\% improvement in planning performance compared with UniV2X, showcasing the strength of our MoE-enhanced multi-level cooperative paradigm.

![framework](./assets/framework.png)

## Installation

1. Create a conda virtual environment.

```
conda create -n unimmv2x python=3.8
conda activate unimmv2x
```

2. Install pytorch and torchvision.

```
conda install cudatoolkit=11.3.1 -c conda-forge
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.1
```

3. Install mmcv-full, mmdet and mmseg.

```
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

4. Install mmdet3d from source code.

```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -v -e .
```

5. Install other requirements.

```
cd ..
git clone https://github.com/Souig/UniMM-V2X.git
cd UniMM-V2X
pip install -r requirements.txt
```

## Dataset Preparation

> Modified from [UniV2X](https://github.com/AIR-THU/UniV2X/).

1. Download V2X-Seq-SPD from [HERE](https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e).
2. Create new V2X-Seq-SPD.

```
python tools/spd_data_converter/gen_example_data.py
    --input YOUR_V2X-Seq-SPD_ROOT \
    --output ./datasets/V2X-Seq-SPD-New \
    --sequences 'all' \
    --update-label \
    --freq 2
```

3. Generate .pkl files.

```
bash ./tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New vehicle-side
bash ./tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New infrastructure-side
bash ./tools/spd_data_converter/spd_dataset_converter.sh V2X-Seq-SPD-New cooperative
```

## Training

```
# infrastructure
# stage1: perception
bash ./tools/unimmv2x_dist_train.sh ./projects/configs_e2e_unimmv2x/unimmv2x_sub_inf_stg1.py ${GPU_NUM}
# stage2: perception + prediction
bash ./tools/unimmv2x_dist_train.sh ./projects/configs_e2e_unimmv2x/unimmv2x_sub_inf_stg2.py ${GPU_NUM}

# vehicle
# stage1: perception
bash ./tools/unimmv2x_dist_train.sh ./projects/configs_e2e_unimmv2x/unimmv2x_sub_veh_stg1.py ${GPU_NUM}
# stage2: perception + prediction + planning
bash ./tools/unimmv2x_dist_train.sh ./projects/configs_e2e_unimmv2x/unimmv2x_sub_veh_stg2.py ${GPU_NUM}

# cooperative training
# stage1
bash ./tools/unimmv2x_dist_train.sh ./projects/configs_e2e_unimmv2x/unimmv2x_coop_stg1.py ${GPU_NUM}
# stage2
bash ./tools/unimmv2x_dist_train.sh ./projects/configs_e2e_unimmv2x/unimmv2x_sub_coop_stg2.py ${GPU_NUM}
```

## Evaluation

Cooperative perception

```
bash ./tools/unimmv2x_dist_eval.sh ./projects/configs_e2e_unimmv2x/unimmv2x_coop_stg1.py ./ckpts/unimmv2x_e2e_stg1.pth ${GPU_NUM}
```

Cooperative planning

```
bash ./tools/unimmv2x_dist_eval.sh ./projects/configs_e2e_unimmv2x/unimmv2x_coop_stg2.py ./ckpts/unimmv2x_e2e_stg2.pth ${GPU_NUM}
```

