# BeTop 项目完整指南

## 目录
1. [项目概述](#项目概述)
2. [环境配置与安装](#环境配置与安装)
3. [数据准备](#数据准备)
4. [模型训练](#模型训练)
5. [模型评估与测试](#模型评估与测试)
6. [迁移到MetaDrive](#迁移到metadrive)
7. [训练数据要求](#训练数据要求)
8. [在MetaDrive上进行轨迹预测](#在metadrive上进行轨迹预测)

---

## 项目概述

### 什么是BeTop？

**BeTop（Behavioral Topology）** 是一个用于自动驾驶场景中多智能体行为建模的创新框架，基于辫理论（Braid Theory）对多智能体未来行为进行推理。

**核心特点：**
- 🎯 利用拓扑结构对多智能体交互进行建模
- 🚗 支持轨迹预测和规划任务
- 📊 在Waymo Open Motion Dataset (WOMD)上实现完整的预测流程
- 🏆 BeTop-ens在2025年WOMD交互预测挑战赛中获得第三名

### 项目架构

**BeTopNet** 是一个协同框架，将拓扑推理与预测任务集成：

```
输入场景数据
    ↓
编码器 (MTR Encoder)
    ↓
拓扑推理
    ↓
解码器 (BeTop Decoder)
    ↓
轨迹预测输出
```

**主要组件：**
- **编码器**: 使用MTR编码器处理智能体和地图特征
- **解码器**: BeTop解码器进行拓扑推理和轨迹预测
- **拓扑建模**: 基于辫理论的多智能体交互建模

---

## 环境配置与安装

### 1. 系统要求

- **操作系统**: Linux (推荐 Ubuntu 18.04+)
- **Python**: 3.9 【womd/betopnet/ops/attention/attention_cuda.cpython-39-x86_64-linux-gnu.so 要求用3.9】
- **CUDA**: 11.3+
- **GPU**: 至少一块支持CUDA的GPU (推荐 A100 80GB)

### 2. 依赖安装

#### 基础依赖
```bash
# 核心依赖包
numpy==1.22.0
tensorflow==2.12.0
torch==1.12.0+cu113
waymo-open-dataset-tf-2-12-0==1.6.4
```

#### 安装步骤

```bash
# 1. 克隆项目
cd /your/project/path

# 2. 安装BeTopNet包
cd womd
pip install -e .

# 3. 编译CUDA扩展
# 参考 EQNet: https://github.com/dvlab-research/DeepVision3D/tree/master/EQNet/eqnet/ops
```

### 3. CUDA扩展编译

项目包含以下CUDA扩展：
- **KNN模块**: K近邻搜索
- **Attention模块**: 自定义注意力计算
- **Grouping模块**: 点云分组操作

安装过程中会自动编译这些扩展，确保：
- CUDA工具包已正确安装
- PyTorch与CUDA版本匹配
- 有足够的编译权限

---

## 数据准备

### 1. 数据下载

#### Waymo Open Motion Dataset (WOMD)

从官方链接下载数据集的 `scenario/` 部分：
- 官方下载地址: [Waymo Open Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_3_0)

**数据集版本选择：**
- **Motion Prediction**: 用于单一智能体轨迹预测
- **Interaction Prediction**: 用于多智能体交互预测

**数据集结构：**
```
waymo_open_dataset_motion_v_1_3_0/
├── training/           # 训练集
├── validation/         # 验证集
├── testing/            # 测试集
└── testing_interactive/  # 交互测试集
```

#### 意图点数据

下载预计算的意图点文件 `intention_points.pkl`：
- 下载地址: [BeTop Releases](https://github.com/OpenDriveLab/BeTop/releases/tag/womd)
- 文件: `cluster_64_center_dict.pkl`

### 2. 数据预处理

#### 原始数据处理

```bash
cd womd/tools/data_tools

# 运行预处理脚本
python3 data_preprocess.py \
    /media/jxy/HIKSEMI/dataset/Waymo1.2.0/betop \
    /media/jxy/G/a_baseline/BeTop/data
```

**预处理功能：**
- 提取场景信息（scenario_id, timestamps等）
- 处理智能体轨迹数据（历史+未来）
- 提取地图多段线（polylines）数据
- 生成训练所需的info文件（.pkl格式）

**生成的文件：**
```
processed_info_path/
├── processed_scenarios_training/
│   ├── sample_xxx.pkl
│   └── ...
├── processed_scenarios_validation/
├── processed_scenarios_training_infos.pkl
└── processed_scenarios_val_infos.pkl
```

### 3. 数据缓存（可选但推荐）

为了提高训练效率和内存利用率，可以预缓存数据为 `.npz` 格式：

```bash
cd womd/tools/data_tools

python3 cache_offline_data.py \
    --cache_path /media/jxy/G/a_baseline/BeTop/cache \
    --cfg /media/jxy/G/a_baseline/BeTop/womd/tools/cfg/BeTopNet_full_64.yaml
```

**注意事项：**
- 缓存过程需要 3-4TB 的存储空间
- 缓存后可以使用更大的batch size
- 训练速度会显著提升

### 4. 配置文件设置

编辑配置文件 `womd/tools/cfg/BeTopNet_full_64.yaml`：

```yaml
DATA_CONFIG:
    DATA_ROOT: '/path/to/your/data/root'
    TRAIN_NPZ_DIR: '/path/to/cached/data'  # 如果使用缓存
    
    SPLIT_DIR: {
        'train': 'processed_scenarios_training', 
        'eval': 'processed_scenarios_validation',
        'test': 'processed_scenarios_testing'
    }
    
    INFO_FILE: {
        'train': 'processed_scenarios_training_infos.pkl', 
        'eval': 'processed_scenarios_val_infos.pkl',
        'test': 'processed_scenarios_test_infos.pkl'
    }

MODEL:
    DECODER:
        INTENTION_POINTS_FILE: '/path/to/cluster_64_center_dict.pkl'
```

---

## 模型训练

### 1. 选择基线模型

BeTop项目支持多个基线模型，配置文件位于 `womd/tools/cfg/`：

| 模型 | 配置文件 | 描述 |
|------|---------|------|
| **BeTopNet-full** | `BeTopNet_full_64.yaml` | 完整版BeTop，使用64个意图点 |
| **BeTopNet-e2e** | `BeTopNet_e2e_6.yaml` | 端到端版本，6个模式 |
| **MTR++** | `MTR_PlusPlus.yaml` | MTR++基线 |
| **Wayformer** | `Wayformer.yaml` | Wayformer基线 |

### 2. 训练命令

#### 单GPU训练
```bash
cd womd/tools

python train.py \
    --cfg_file cfg/BeTopNet_full_64.yaml \
    --batch_size 10 \
    --epochs 30 \
    --extra_tag experiment_name
```

#### 多GPU分布式训练（推荐）
```bash
cd womd/tools

bash scripts/dist_train.sh 4 \
    --cfg_file cfg/BeTopNet_full_64.yaml \
    --epoch 30 \
    --batch_size 40 \
    --extra_tag multi_gpu_training
```

**参数说明：**
- `N_GPUS`: 使用的GPU数量（如4表示4块GPU）
- `--cfg_file`: 模型配置文件路径
- `--epoch`: 训练轮数（推荐30）
- `--batch_size`: 总批量大小
- `--extra_tag`: 实验标识名称

### 3. 训练配置详解

#### 批量大小建议
- **未缓存数据**: `BATCH_SIZE = 10 * N_GPUS` (使用A100 80G)
- **已缓存数据**: 可以使用更大的batch size（如16 * N_GPUS）

#### 优化器配置
```yaml
OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 10
    NUM_EPOCHS: 30
    
    OPTIMIZER: AdamW
    LR: 0.0001              # 学习率
    WEIGHT_DECAY: 0.01      # 权重衰减
    
    SCHEDULER: lambdaLR
    DECAY_STEP_LIST: [22, 24, 26, 28]  # 学习率衰减步骤
    LR_DECAY: 0.5           # 衰减率
    LR_CLIP: 0.000001       # 最小学习率
```

#### 损失函数权重
```yaml
DECODER:
    LOSS_WEIGHTS: {
        'cls': 1.0,    # 分类损失
        'reg': 1.0,    # 回归损失
        'vel': 0.5,    # 速度损失
        'top': 100     # 拓扑损失（BeTop特有）
    }
```

### 4. 训练监控

训练过程中，日志和检查点会保存在：
```
womd/output/cfg/BeTopNet_full_64/experiment_name/
├── log_train_xxx.txt           # 训练日志
├── tensorboard/                # TensorBoard日志
├── ckpt/                       # 模型检查点
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_2.pth
│   └── ...
└── eval/                       # 评估结果
```

使用TensorBoard查看训练过程：
```bash
tensorboard --logdir=womd/output/cfg/BeTopNet_full_64/experiment_name/tensorboard
```

---

## 模型评估与测试

### 1. 模型评估

在验证集上评估训练好的模型：

```bash
cd womd/tools

bash scripts/dist_test.sh 4 \
    --cfg_file cfg/BeTopNet_full_64.yaml \
    --batch_size 40 \
    --ckpt /path/to/checkpoint_epoch_30.pth
```

**评估指标（WOMD标准）：**
- **minADE**: 最小平均位移误差
- **minFDE**: 最小最终位移误差
- **MR**: 错失率 (Miss Rate)
- **mAP**: 平均精度
- **Soft mAP**: 软平均精度

### 2. 提交到排行榜

#### 配置提交信息

编辑 `womd/tools/submission.py` 第188行：

```python
submission_info = dict(
    account_name='your_waymo_account@email.com',
    unique_method_name='BeTopNet_v1',
    authors=['Your Name', 'Collaborator Name'],
    affiliation='Your University/Company',
    uses_lidar_data=False,
    uses_camera_data=False,
    uses_public_model_pretraining=False,
    public_model_names='N/A',
    num_model_parameters='N/A',
)
```

#### 生成提交文件

**Motion Prediction 提交：**
```bash
cd womd/tools

python3 submission.py \
    --cfg_file cfg/BeTopNet_full_64.yaml \
    --batch_size 40 \
    --ckpt /path/to/checkpoint.pth \
    --output_dir ./submission_output
```

**Interaction Prediction 提交：**
```bash
python3 submission.py \
    --cfg_file cfg/BeTopNet_full_64.yaml \
    --batch_size 40 \
    --ckpt /path/to/checkpoint.pth \
    --output_dir ./submission_output \
    --interactive
```

**Eval集提交：**
```bash
python3 submission.py \
    --cfg_file cfg/BeTopNet_full_64.yaml \
    --batch_size 40 \
    --ckpt /path/to/checkpoint.pth \
    --output_dir ./submission_output \
    --eval
```

#### 上传结果

生成的 `.tar.gz` 文件位于 `--output_dir` 指定的目录，上传至：
- [Motion Prediction Challenge](https://waymo.com/open/challenges/2024/motion-prediction/)
- [Interaction Prediction Challenge](https://waymo.com/open/challenges/2021/interaction-prediction/)

---

## 迁移到MetaDrive

### 1. MetaDrive简介

**MetaDrive** 是一个用于自动驾驶研究的开源模拟器：
- 支持多样化场景生成
- 提供真实的车辆动力学模型
- 支持多智能体交互仿真
- 可以从真实数据集（如WOMD、nuPlan）导入场景

### 2. 迁移架构设计

#### 整体架构
```
WOMD数据训练 → BeTopNet模型 → MetaDrive适配层 → MetaDrive仿真预测
```

#### 关键模块映射

| BeTop组件 | MetaDrive对应 | 迁移策略 |
|----------|--------------|---------|
| 场景数据 | MetaDrive Scenario | 格式转换 |
| 智能体轨迹 | Vehicle State | 状态映射 |
| 地图数据 | MetaDrive Map | 坐标转换 |
| 预测输出 | Policy/Planner | 接口封装 |

### 3. 数据格式转换

#### WOMD到MetaDrive的场景转换

创建转换脚本 `womd_to_metadrive_converter.py`：

```python
import numpy as np
from metadrive.scenario import ScenarioDescription
from metadrive.type_utils import MetaDriveType

class WOMDToMetaDriveConverter:
    """将WOMD场景转换为MetaDrive格式"""
    
    def __init__(self):
        self.type_mapping = {
            'TYPE_VEHICLE': MetaDriveType.VEHICLE,
            'TYPE_PEDESTRIAN': MetaDriveType.PEDESTRIAN,
            'TYPE_CYCLIST': MetaDriveType.CYCLIST,
        }
    
    def convert_scenario(self, womd_data):
        """
        转换WOMD场景数据到MetaDrive格式
        
        Args:
            womd_data: WOMD数据字典，包含：
                - scenario_id: 场景ID
                - track_infos: 轨迹信息
                - map_infos: 地图信息
                - timestamps: 时间戳
        
        Returns:
            MetaDrive场景描述对象
        """
        scenario = ScenarioDescription()
        
        # 1. 转换场景基本信息
        scenario.scenario_id = womd_data['scenario_id']
        scenario.time_step = 0.1  # WOMD采样率10Hz
        
        # 2. 转换智能体轨迹
        for track_id, track_info in enumerate(womd_data['track_infos']):
            agent = {
                'type': self.type_mapping.get(track_info['object_type']),
                'state': self._convert_trajectory(track_info['trajs']),
                'id': track_info['object_id']
            }
            scenario.add_agent(agent)
        
        # 3. 转换地图数据
        map_features = self._convert_map(womd_data['map_infos'])
        scenario.map = map_features
        
        return scenario
    
    def _convert_trajectory(self, trajs):
        """
        转换轨迹格式
        WOMD: [x, y, z, length, width, height, heading, vx, vy, valid]
        MetaDrive: [x, y, heading, velocity, ...]
        """
        converted = []
        for traj in trajs:
            if traj[-1] == 1:  # valid flag
                state = {
                    'position': [traj[0], traj[1]],
                    'heading': traj[6],
                    'velocity': np.sqrt(traj[7]**2 + traj[8]**2),
                    'length': traj[3],
                    'width': traj[4]
                }
                converted.append(state)
        return converted
    
    def _convert_map(self, map_infos):
        """转换地图polylines到MetaDrive格式"""
        map_features = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'crosswalk': []
        }
        
        for polyline in map_infos['all_polylines']:
            feature_type = polyline['type']
            points = polyline['polyline']
            
            # 根据类型分类存储
            if 'LANE' in feature_type:
                map_features['lane'].append(points)
            elif 'LINE' in feature_type:
                map_features['road_line'].append(points)
            elif 'EDGE' in feature_type:
                map_features['road_edge'].append(points)
            elif 'CROSSWALK' in feature_type:
                map_features['crosswalk'].append(points)
        
        return map_features

# 使用示例
converter = WOMDToMetaDriveConverter()
metadrive_scenario = converter.convert_scenario(womd_data)
```

### 4. 模型接口封装

#### BeTopNet预测器封装

创建 `betop_metadrive_predictor.py`：

```python
import torch
from metadrive.policy.base_policy import BasePolicy
from betopnet.models import build_model
from betopnet.config import cfg

class BeTopMetaDrivePredictor(BasePolicy):
    """将BeTopNet封装为MetaDrive预测器"""
    
    def __init__(self, checkpoint_path, config_path, device='cuda'):
        super().__init__()
        
        # 加载BeTop模型
        self.device = device
        self.model = self._load_model(checkpoint_path, config_path)
        self.model.eval()
        
    def _load_model(self, checkpoint_path, config_path):
        """加载训练好的BeTopNet模型"""
        from betopnet.config import cfg_from_yaml_file
        
        cfg_from_yaml_file(config_path, cfg)
        model = build_model(cfg.MODEL)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def act(self, obs):
        """
        MetaDrive标准接口：根据观测预测行为
        
        Args:
            obs: MetaDrive观测字典，包含：
                - agents: 周围智能体信息
                - map: 地图信息
                - ego: 自车状态
        
        Returns:
            预测的未来轨迹
        """
        # 1. 转换MetaDrive观测为BeTop输入格式
        batch_dict = self._prepare_input(obs)
        
        # 2. 模型推理
        with torch.no_grad():
            predictions = self.model(batch_dict)
        
        # 3. 提取并转换预测结果
        trajectories = self._extract_predictions(predictions)
        
        return trajectories
    
    def _prepare_input(self, obs):
        """将MetaDrive观测转换为BeTop输入格式"""
        batch_dict = {}
        
        # 转换智能体数据
        obj_trajs = []
        obj_trajs_mask = []
        
        for agent_id, agent_state in obs['agents'].items():
            # 提取历史轨迹 (过去11帧，1.1秒)
            history = agent_state['history']  # List of states
            traj = self._format_trajectory(history)
            obj_trajs.append(traj)
            obj_trajs_mask.append([1] * len(traj))
        
        batch_dict['obj_trajs'] = torch.tensor(
            obj_trajs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        batch_dict['obj_trajs_mask'] = torch.tensor(
            obj_trajs_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        
        # 转换地图数据
        map_polylines = self._format_map(obs['map'])
        batch_dict['map_polylines'] = torch.tensor(
            map_polylines, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        # 其他必要字段
        batch_dict['obj_trajs_pos'] = batch_dict['obj_trajs'][..., :3]
        batch_dict['center_objects_world'] = obs['ego']['position']
        
        return batch_dict
    
    def _format_trajectory(self, history):
        """格式化轨迹数据为BeTop输入格式"""
        traj = []
        for state in history:
            # BeTop输入: [x, y, z, length, width, height, heading, 
            #             vx, vy, ax, ay, ...] (29维)
            features = [
                state['position'][0], state['position'][1], 0,  # x, y, z
                state['length'], state['width'], 1.5,  # dimensions
                state['heading'],  # heading
                state['velocity'] * np.cos(state['heading']),  # vx
                state['velocity'] * np.sin(state['heading']),  # vy
                # ... 其他特征可以补零或从状态计算
            ]
            # 填充到29维
            features.extend([0] * (29 - len(features)))
            traj.append(features)
        return traj
    
    def _format_map(self, map_data):
        """格式化地图数据"""
        polylines = []
        
        for lane in map_data['lanes']:
            # 采样20个点
            sampled = self._sample_polyline(lane['points'], 20)
            # 格式化为 [x, y, z, dx, dy, type_onehot...] (9维)
            formatted = self._format_polyline(sampled, lane['type'])
            polylines.append(formatted)
        
        # 填充或截断到768条polylines
        return self._pad_polylines(polylines, 768)
    
    def _extract_predictions(self, predictions):
        """从模型输出提取预测轨迹"""
        # BeTop输出格式：
        # - pred_scores: (B, N, K) - K个模式的概率
        # - pred_trajs: (B, N, K, T, 2) - K个模式的轨迹
        
        pred_scores = predictions['pred_scores'][0]  # (N, K)
        pred_trajs = predictions['pred_trajs'][0]    # (N, K, T, 2)
        
        # 选择最可能的模式
        best_mode = torch.argmax(pred_scores, dim=-1)  # (N,)
        
        trajectories = {}
        for i, mode_idx in enumerate(best_mode):
            agent_id = predictions['track_index_to_predict'][i]
            traj = pred_trajs[i, mode_idx].cpu().numpy()  # (T, 2)
            trajectories[agent_id] = traj
        
        return trajectories

# 使用示例
predictor = BeTopMetaDrivePredictor(
    checkpoint_path='path/to/checkpoint_epoch_30.pth',
    config_path='path/to/BeTopNet_full_64.yaml'
)
```

### 5. MetaDrive集成

#### 创建完整的预测流程

创建 `metadrive_betop_integration.py`：

```python
from metadrive import MetaDriveEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from betop_metadrive_predictor import BeTopMetaDrivePredictor

class BeTopMetaDriveEnv:
    """集成BeTop预测器的MetaDrive环境"""
    
    def __init__(self, config):
        # 初始化MetaDrive环境
        self.env = MetaDriveEnv(config)
        
        # 初始化BeTop预测器
        self.predictor = BeTopMetaDrivePredictor(
            checkpoint_path=config['betop_checkpoint'],
            config_path=config['betop_config']
        )
        
    def run_prediction(self, scenario_id=None):
        """运行轨迹预测"""
        # 重置环境
        obs = self.env.reset(scenario_id=scenario_id)
        
        predictions_history = []
        ground_truth_history = []
        
        done = False
        while not done:
            # 使用BeTop预测未来轨迹
            predictions = self.predictor.act(obs)
            predictions_history.append(predictions)
            
            # 环境步进（使用replay policy或其他控制策略）
            obs, reward, done, info = self.env.step(action=None)
            
            # 记录真实轨迹用于评估
            ground_truth = self._extract_ground_truth(info)
            ground_truth_history.append(ground_truth)
        
        # 评估预测结果
        metrics = self._evaluate_predictions(
            predictions_history, ground_truth_history
        )
        
        return {
            'predictions': predictions_history,
            'ground_truth': ground_truth_history,
            'metrics': metrics
        }
    
    def _extract_ground_truth(self, info):
        """从环境信息中提取真实轨迹"""
        gt = {}
        for agent_id, agent in self.env.vehicles.items():
            gt[agent_id] = {
                'position': agent.position,
                'heading': agent.heading,
                'velocity': agent.velocity
            }
        return gt
    
    def _evaluate_predictions(self, predictions, ground_truth):
        """评估预测指标"""
        metrics = {
            'ADE': [],  # Average Displacement Error
            'FDE': [],  # Final Displacement Error
        }
        
        # 计算逐帧误差
        for pred, gt in zip(predictions, ground_truth):
            for agent_id in pred.keys():
                if agent_id in gt:
                    pred_traj = pred[agent_id]
                    gt_pos = gt[agent_id]['position']
                    
                    # 计算位移误差
                    displacement = np.linalg.norm(
                        pred_traj[0] - gt_pos
                    )
                    metrics['ADE'].append(displacement)
        
        # 聚合指标
        return {
            'ADE': np.mean(metrics['ADE']),
            'FDE': np.mean(metrics['FDE']),
        }

# 使用示例
config = {
    'use_render': True,
    'manual_control': False,
    'traffic_density': 0.3,
    'start_scenario_index': 0,
    'num_scenarios': 100,
    'betop_checkpoint': 'path/to/checkpoint.pth',
    'betop_config': 'path/to/config.yaml',
    'data_directory': 'path/to/womd/scenarios',  # 可以导入WOMD场景
}

env = BeTopMetaDriveEnv(config)
results = env.run_prediction(scenario_id='scenario_001')

print(f"ADE: {results['metrics']['ADE']:.3f}m")
print(f"FDE: {results['metrics']['FDE']:.3f}m")
```

### 6. 批量场景测试

创建 `batch_test_metadrive.py`：

```python
import os
import json
from tqdm import tqdm
from metadrive_betop_integration import BeTopMetaDriveEnv

def batch_test_scenarios(scenario_dir, checkpoint_path, config_path, output_dir):
    """批量测试多个场景"""
    
    # 获取所有场景ID
    scenario_files = os.listdir(scenario_dir)
    scenario_ids = [f.replace('.pkl', '').replace('sample_', '') 
                   for f in scenario_files if f.endswith('.pkl')]
    
    # 初始化环境
    config = {
        'betop_checkpoint': checkpoint_path,
        'betop_config': config_path,
        'data_directory': scenario_dir,
        'use_render': False,  # 批量测试时关闭渲染
    }
    env = BeTopMetaDriveEnv(config)
    
    # 运行测试
    all_results = []
    for scenario_id in tqdm(scenario_ids, desc="Testing scenarios"):
        try:
            result = env.run_prediction(scenario_id=scenario_id)
            all_results.append({
                'scenario_id': scenario_id,
                'metrics': result['metrics']
            })
        except Exception as e:
            print(f"Error in scenario {scenario_id}: {e}")
            continue
    
    # 汇总结果
    avg_metrics = {
        'ADE': np.mean([r['metrics']['ADE'] for r in all_results]),
        'FDE': np.mean([r['metrics']['FDE'] for r in all_results]),
    }
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'summary': avg_metrics,
            'details': all_results
        }, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Total scenarios tested: {len(all_results)}")
    print(f"Average ADE: {avg_metrics['ADE']:.3f}m")
    print(f"Average FDE: {avg_metrics['FDE']:.3f}m")
    
    return all_results

# 运行批量测试
results = batch_test_scenarios(
    scenario_dir='/path/to/processed_scenarios_validation',
    checkpoint_path='/path/to/checkpoint_epoch_30.pth',
    config_path='/path/to/BeTopNet_full_64.yaml',
    output_dir='./metadrive_test_results'
)
```

---

## 训练数据要求

### 1. WOMD数据集详细要求

#### 数据集规模
- **训练集**: ~487K 场景
- **验证集**: ~44K 场景
- **测试集**: ~44K 场景

#### 每个场景包含

**时间范围：**
- 历史: 1秒 (11帧，10Hz采样)
- 未来: 8秒 (80帧，10Hz采样)

**智能体信息：**
```python
{
    'object_id': int,              # 智能体ID
    'object_type': str,            # 类型: VEHICLE/PEDESTRIAN/CYCLIST
    'trajs': np.ndarray,           # (91, 10) - 轨迹数据
                                   # [x, y, z, length, width, height, 
                                   #  heading, vx, vy, valid]
    'valid_mask': np.ndarray,      # (91,) - 有效性掩码
}
```

**地图信息：**
```python
{
    'polylines': List[Dict],       # 多段线列表
    'polyline_type': str,          # 类型: LANE_CENTER/ROAD_EDGE/
                                   #       STOP_SIGN/CROSSWALK/等
    'polyline_points': np.ndarray, # (N, 7) - 点坐标和属性
}
```

**场景元数据：**
- `scenario_id`: 唯一场景标识符
- `timestamps_seconds`: 时间戳数组
- `current_time_index`: 当前时刻索引（通常是10）
- `sdc_track_index`: 自车轨迹索引
- `tracks_to_predict`: 需要预测的智能体列表

### 2. 数据增强策略

BeTop使用以下数据增强方法：

#### 空间增强
- **随机旋转**: 场景整体旋转 [-π, π]
- **随机平移**: 小范围平移 ±2米
- **坐标归一化**: 以预测目标为中心

#### 时间增强
- **随机时间偏移**: 改变历史观测的起始时刻
- **轨迹采样**: 不同的采样间隔

### 3. 特征工程

#### 智能体特征（29维）
```python
agent_features = [
    # 位置 (3维)
    x, y, z,
    
    # 尺寸 (3维)
    length, width, height,
    
    # 运动状态 (5维)
    heading, velocity_x, velocity_y, 
    acceleration_x, acceleration_y,
    
    # 相对特征 (6维)
    relative_x, relative_y, relative_heading,
    distance_to_center, angle_to_center,
    time_to_collision,
    
    # 类型编码 (one-hot, 3维)
    is_vehicle, is_pedestrian, is_cyclist,
    
    # 其他属性 (9维)
    lane_id, speed, acceleration, jerk,
    curvature, is_valid, is_predicted, 
    is_sdc, timestamp
]
```

#### 地图特征（9维）
```python
map_features = [
    # 位置 (3维)
    x, y, z,
    
    # 方向 (2维)
    direction_x, direction_y,
    
    # 类型编码 (one-hot, 4维)
    is_lane, is_road_line, is_road_edge, is_crosswalk
]
```

### 4. 训练集统计信息

| 统计项 | 数值 |
|--------|------|
| 场景总数 | 487,000+ |
| 智能体总数 | ~8M |
| 平均每场景智能体数 | 16.4 |
| 地图Polyline总数 | ~100M |
| 平均每场景Polyline数 | 205.7 |
| 数据集总大小 | ~1TB (原始) |
| 预处理后大小 | ~1.5TB |
| 缓存后大小 | ~3-4TB |

### 5. 训练数据质量要求

#### 必要的质量过滤
- 智能体至少有5帧有效历史轨迹
- 未来轨迹至少有30帧有效数据
- 场景包含至少1个需要预测的智能体
- 地图数据完整且有效

#### 数据验证
```python
def validate_scenario(scenario):
    """验证场景数据质量"""
    checks = {
        'has_valid_tracks': len(scenario['tracks_to_predict']) > 0,
        'has_sufficient_history': scenario['current_time_index'] >= 5,
        'has_future_gt': scenario['num_future_frames'] >= 30,
        'has_map': len(scenario['map_infos']['all_polylines']) > 0,
        'valid_timestamps': len(scenario['timestamps']) == 91,
    }
    return all(checks.values())
```

---

## 在MetaDrive上进行轨迹预测

### 1. 完整工作流程

```
步骤1: 数据准备
    ↓
步骤2: 场景导入MetaDrive
    ↓
步骤3: BeTop模型加载
    ↓
步骤4: 实时预测
    ↓
步骤5: 评估与可视化
```

### 2. 实时预测流程

#### 主预测循环

```python
class RealtimePredictor:
    """实时轨迹预测器"""
    
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.prediction_horizon = 80  # 8秒，80帧
        self.update_frequency = 10    # 10Hz
        
    def run(self):
        """运行实时预测"""
        obs = self.env.reset()
        done = False
        
        while not done:
            # 1. 当前观测
            current_obs = self._get_observation()
            
            # 2. 预测未来轨迹
            predictions = self.predictor.act(current_obs)
            
            # 3. 可视化预测
            self._visualize_predictions(predictions)
            
            # 4. 环境步进
            action = self._get_action(predictions)
            obs, reward, done, info = self.env.step(action)
            
            # 5. 评估预测精度
            metrics = self._evaluate_step(predictions, obs)
            
        return metrics
    
    def _get_observation(self):
        """获取当前观测"""
        obs = {
            'ego': self.env.vehicle.get_state(),
            'agents': {},
            'map': {}
        }
        
        # 收集周围智能体
        for v_id, vehicle in self.env.vehicles.items():
            if v_id != self.env.vehicle.id:
                obs['agents'][v_id] = {
                    'position': vehicle.position,
                    'heading': vehicle.heading,
                    'velocity': vehicle.velocity,
                    'history': vehicle.get_history()
                }
        
        # 收集地图信息
        obs['map'] = self.env.current_map.get_map_features(
            center=self.env.vehicle.position,
            radius=100  # 100米范围
        )
        
        return obs
    
    def _visualize_predictions(self, predictions):
        """可视化预测轨迹"""
        if self.env.config['use_render']:
            for agent_id, traj in predictions.items():
                # 在MetaDrive中绘制预测轨迹
                self.env.render_trajectory(
                    traj, 
                    color='red',
                    width=2
                )
    
    def _evaluate_step(self, predictions, next_obs):
        """评估单步预测精度"""
        errors = []
        for agent_id, pred_traj in predictions.items():
            if agent_id in next_obs['agents']:
                gt_pos = next_obs['agents'][agent_id]['position']
                pred_pos = pred_traj[0]  # 第一帧预测
                error = np.linalg.norm(pred_pos - gt_pos)
                errors.append(error)
        
        return {
            'step_ADE': np.mean(errors) if errors else 0
        }
```

### 3. 完整示例代码

创建 `run_betop_in_metadrive.py`：

```python
#!/usr/bin/env python3
"""
BeTop在MetaDrive中运行的完整示例
"""

import numpy as np
import argparse
from metadrive import MetaDriveEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.scenario_description import ScenarioDescription

from betop_metadrive_predictor import BeTopMetaDrivePredictor
from womd_to_metadrive_converter import WOMDToMetaDriveConverter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to BeTop checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to BeTop config file')
    parser.add_argument('--scenario_dir', type=str, required=True,
                       help='Directory containing WOMD scenarios')
    parser.add_argument('--render', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--num_scenarios', type=int, default=10,
                       help='Number of scenarios to test')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 初始化MetaDrive环境
    print("Initializing MetaDrive environment...")
    env_config = {
        'use_render': args.render,
        'manual_control': False,
        'data_directory': args.scenario_dir,
        'num_scenarios': args.num_scenarios,
        'start_scenario_index': 0,
        'horizon': 1000,
    }
    env = MetaDriveEnv(env_config)
    
    # 2. 初始化BeTop预测器
    print("Loading BeTop model...")
    predictor = BeTopMetaDrivePredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    # 3. 初始化转换器
    converter = WOMDToMetaDriveConverter()
    
    # 4. 运行测试
    print(f"Testing on {args.num_scenarios} scenarios...")
    all_metrics = []
    
    for scenario_idx in range(args.num_scenarios):
        print(f"\nScenario {scenario_idx + 1}/{args.num_scenarios}")
        
        # 重置环境
        obs = env.reset(scenario_index=scenario_idx)
        done = False
        step = 0
        
        scenario_metrics = {
            'ADE': [],
            'FDE': [],
            'step_errors': []
        }
        
        # 运行场景
        while not done and step < 80:  # 8秒预测
            # 预测
            predictions = predictor.act(obs)
            
            # 环境步进
            obs, reward, done, info = env.step(None)  # Replay模式
            
            # 评估
            if step == 0:  # 只在第一帧评估完整轨迹
                # 收集未来80帧的真实轨迹
                future_gt = []
                temp_obs = obs
                for _ in range(80):
                    future_gt.append(temp_obs['agents'])
                    temp_obs, _, _, _ = env.step(None)
                
                # 计算ADE和FDE
                for agent_id, pred_traj in predictions.items():
                    if agent_id in future_gt[0]:
                        # ADE: 平均位移误差
                        ade = 0
                        valid_frames = 0
                        for t, gt_frame in enumerate(future_gt):
                            if agent_id in gt_frame and t < len(pred_traj):
                                gt_pos = gt_frame[agent_id]['position']
                                pred_pos = pred_traj[t]
                                ade += np.linalg.norm(pred_pos - gt_pos)
                                valid_frames += 1
                        
                        if valid_frames > 0:
                            ade /= valid_frames
                            scenario_metrics['ADE'].append(ade)
                            
                            # FDE: 最终位移误差
                            if agent_id in future_gt[-1]:
                                fde = np.linalg.norm(
                                    pred_traj[-1] - future_gt[-1][agent_id]['position']
                                )
                                scenario_metrics['FDE'].append(fde)
            
            step += 1
        
        # 汇总场景指标
        scenario_summary = {
            'scenario_id': scenario_idx,
            'ADE': np.mean(scenario_metrics['ADE']) if scenario_metrics['ADE'] else 0,
            'FDE': np.mean(scenario_metrics['FDE']) if scenario_metrics['FDE'] else 0,
        }
        all_metrics.append(scenario_summary)
        
        print(f"  ADE: {scenario_summary['ADE']:.3f}m")
        print(f"  FDE: {scenario_summary['FDE']:.3f}m")
    
    # 5. 输出总体结果
    print("\n" + "="*50)
    print("OVERALL RESULTS")
    print("="*50)
    avg_ade = np.mean([m['ADE'] for m in all_metrics])
    avg_fde = np.mean([m['FDE'] for m in all_metrics])
    print(f"Average ADE: {avg_ade:.3f}m")
    print(f"Average FDE: {avg_fde:.3f}m")
    print(f"Total scenarios: {len(all_metrics)}")
    
    # 保存结果
    import json
    with open('metadrive_results.json', 'w') as f:
        json.dump({
            'summary': {
                'avg_ADE': avg_ade,
                'avg_FDE': avg_fde,
                'num_scenarios': len(all_metrics)
            },
            'details': all_metrics
        }, f, indent=2)
    
    print("\nResults saved to metadrive_results.json")
    
    env.close()

if __name__ == '__main__':
    main()
```

### 4. 运行预测

```bash
# 基本运行
python run_betop_in_metadrive.py \
    --checkpoint /path/to/checkpoint_epoch_30.pth \
    --config /path/to/BeTopNet_full_64.yaml \
    --scenario_dir /path/to/processed_scenarios_validation \
    --num_scenarios 100

# 带可视化运行
python run_betop_in_metadrive.py \
    --checkpoint /path/to/checkpoint_epoch_30.pth \
    --config /path/to/BeTopNet_full_64.yaml \
    --scenario_dir /path/to/processed_scenarios_validation \
    --num_scenarios 10 \
    --render
```

### 5. 性能优化建议

#### 推理加速
```python
# 使用半精度推理
model.half()
batch_dict = {k: v.half() if torch.is_tensor(v) else v 
              for k, v in batch_dict.items()}

# 使用TorchScript
scripted_model = torch.jit.script(model)

# 批量预测多个智能体
# 将多个智能体组batch处理而不是逐个处理
```

#### 内存优化
```python
# 限制场景缓存大小
max_cache_size = 100

# 使用生成器加载数据
def scenario_generator(scenario_dir):
    for scenario_file in os.listdir(scenario_dir):
        yield load_scenario(scenario_file)
        
# 清理GPU缓存
torch.cuda.empty_cache()
```

### 6. 常见问题与解决方案

#### Q1: 坐标系不匹配
**问题**: WOMD和MetaDrive使用不同的坐标系统

**解决方案**:
```python
def convert_coordinates(womd_pos, womd_heading):
    """转换WOMD坐标到MetaDrive坐标"""
    # WOMD: x右，y前，z上
    # MetaDrive: x前，y右，z上
    metadrive_pos = [womd_pos[1], womd_pos[0], womd_pos[2]]
    metadrive_heading = womd_heading + np.pi/2
    return metadrive_pos, metadrive_heading
```

#### Q2: 预测延迟过高
**问题**: 实时预测时FPS过低

**解决方案**:
- 减少预测频率（如5Hz而非10Hz）
- 使用模型蒸馏得到更快的模型
- 使用TensorRT进行加速

#### Q3: 地图数据缺失
**问题**: 某些场景地图信息不完整

**解决方案**:
```python
def fill_missing_map(scenario):
    """填充缺失的地图数据"""
    if len(scenario['map_infos']['all_polylines']) == 0:
        # 使用最近场景的地图或生成默认地图
        scenario['map_infos'] = get_default_map()
    return scenario
```

---

## 总结

### 关键步骤回顾

1. **环境安装**: 配置CUDA、PyTorch、WOMD工具包
2. **数据准备**: 下载WOMD数据，预处理，可选缓存
3. **模型训练**: 使用BeTopNet在WOMD上训练，30个epoch
4. **模型评估**: 在验证集上测试，计算mAP、ADE、FDE等指标
5. **MetaDrive迁移**:
   - 实现数据格式转换器
   - 封装BeTop预测器接口
   - 集成到MetaDrive环境
6. **轨迹预测**: 在MetaDrive中运行实时预测和评估

### 预期性能

基于BeTop论文和WOMD挑战赛结果：

| 指标 | BeTopNet-full | BeTop-ens |
|------|--------------|-----------|
| Soft mAP | ~0.40 | ~0.45 |
| minADE (m) | ~1.8 | ~1.6 |
| minFDE (m) | ~3.5 | ~3.0 |
| Miss Rate | ~0.15 | ~0.12 |

### 进一步优化方向

1. **模型集成**: 训练多个模型进行集成，如BeTop-ens
2. **后处理**: 添加物理约束、碰撞检测、可行性过滤
3. **自适应预测**: 根据场景复杂度动态调整预测策略
4. **增量学习**: 使用MetaDrive收集的数据继续训练模型
5. **闭环测试**: 在MetaDrive中进行规划+预测的闭环测试

### 参考资源

- **BeTop论文**: [arXiv:2409.18031](https://arxiv.org/abs/2409.18031)
- **WOMD官网**: [waymo.com/open](https://waymo.com/open/)
- **MetaDrive文档**: [metadrive-simulator.readthedocs.io](https://metadrive-simulator.readthedocs.io/)
- **BeTop GitHub**: [github.com/OpenDriveLab/BeTop](https://github.com/OpenDriveLab/BeTop)

---

## 附录

### A. 配置文件模板

完整的 `BeTopNet_full_64.yaml` 配置示例参见项目文件。

### B. 常用命令速查

```bash
# 训练
bash womd/tools/scripts/dist_train.sh 4 --cfg_file cfg/BeTopNet_full_64.yaml --epoch 30 --batch_size 40

# 评估
bash womd/tools/scripts/dist_test.sh 4 --cfg_file cfg/BeTopNet_full_64.yaml --ckpt path/to/ckpt --batch_size 40

# 生成提交文件
python womd/tools/submission.py --cfg_file cfg/BeTopNet_full_64.yaml --ckpt path/to/ckpt --output_dir ./submission

# MetaDrive预测
python run_betop_in_metadrive.py --checkpoint path/to/ckpt --config cfg/BeTopNet_full_64.yaml --scenario_dir path/to/scenarios --render
```

### C. 故障排查

常见错误及解决方法请参考项目Issues或联系作者。

---

**文档版本**: 1.0  
**最后更新**: 2025年10月  
**作者**: BeTop项目团队  
**联系方式**: haochen002@e.ntu.edu.sg

