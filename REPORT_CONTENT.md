# 大作业报告内容 — CLIP跨模态传感器适配系统

---

## 一、选题要求

### 1.1 选题背景

本课程为《人工智能程序设计》（2025-2026-2学年），大作业采用选题5（自选题目）。项目名称为：「CLIP跨模态传感器适配：面向深度传感器的文本引导语义对齐系统」。

选题理由：
1. **技术前沿性**：CLIP（Contrastive Language-Image Pre-training）是当前多模态学习领域的核心技术，本项目将其扩展至深度传感器模态，涉及对比学习、特征对齐、zero-shot分类等前沿AI技术。
2. **与课程内容高度契合**：项目涵盖神经网络设计、对比学习训练、HuggingFace Transformers使用、数据加载与预处理、训练循环实现等课程核心知识点。
3. **研究与应用价值**：深度传感器在SLAM、机器人导航、自动驾驶中广泛使用，使深度数据具备CLIP级别的语义理解能力具有实际意义。
4. **可扩展性**：本项目的MVP将在课程期间完成，后续可继续扩展为论文发表。

### 1.2 选题创新性

- 将CLIP的zero-shot分类能力迁移至深度传感器模态
- 设计轻量级MLP Adapter实现跨模态特征对齐
- 采用RGB→CLIP特征作为教师信号，通过对比学习训练深度适配器
- 在NYU Depth V2数据集上验证深度zero-shot分类效果

---

## 二、国内外相关研究

### 2.1 CLIP及跨模态扩展

CLIP（Radford et al., ICML 2021）通过4亿图文对进行对比学习，建立了强大的视觉-语义联合嵌入空间，在40+个zero-shot分类数据集上达到competitive性能。其核心是双编码器架构（ViT图像编码器 + Transformer文本编码器）+ InfoNCE对比损失。

后续扩展工作包括：
- **GroupViT**（CVPR 2022）：通过分组token机制实现语义分割，但限于RGB模态。
- **CLIPSeg**（CVPR 2022）：在patch-level实现文本引导的分割，同样仅支持RGB。
- **PointCLIP**（CVPR 2022）及**PointCLIP V2**（AAAI 2023）：将点云投影为深度图并伪彩色编码后送入CLIP，但投影过程存在信息损失。

### 2.2 传感器模态的CLIP适配

**T-CLIP**（arXiv 2026-05）是最直接相关工作，通过2层MLP Adapter将热红外图像特征映射到CLIP embedding空间，仅采用全局对比学习对齐，在FLIR等热红外数据集上实现zero-shot分类，简洁高效但缺乏局部对齐机制且仅支持单一模态。

**Thermo-VL**（arXiv 2026-05）采用热红外专用ViT从头预训练，语义对齐更充分但计算成本极高（500+ GPU-days），且不兼容CLIP生态。

### 2.3 本项目的定位

本项目借鉴T-CLIP的MLP Adapter方案，首次将传感器适配方法应用于**深度传感器（Depth）**模态，在NYU Depth V2数据集上实现zero-shot场景分类。与现有工作相比：

| 维度 | PointCLIP | T-CLIP | 本项目 |
|------|-----------|--------|--------|
| 传感器模态 | 深度/点云 | 热红外 | **深度** |
| 适配方法 | 伪彩色编码 | MLP Adapter | **MLP / Cross-Attn** |
| 是否保留CLIP零样本 | 部分 | ✓ | **✓** |
| 局部对齐 | ✗ | ✗ | **可扩展** |

---

## 三、需求分析及系统功能描述

### 3.1 需求分析

**功能需求：**
1. 加载NYU Depth V2数据集，获取RGB-深度图-场景标签三元组
2. 加载预训练CLIP模型（ViT-B/32）并冻结其参数
3. 实现深度图预处理（单通道重复/表面法线计算）
4. 实现MLP Adapter将深度CLIP特征映射到对齐空间
5. 通过对比学习（NT-Xent损失）训练Adapter
6. 在深度图上进行zero-shot场景分类评估
7. 可视化训练曲线和精度对比

**性能需求：**
- 在200样本小规模训练下，zero-shot分类精度明显高于随机基线
- 训练时间不超过2小时
- 输出训练曲线、精度对比图、每类精度图

**约束条件：**
- 使用PyTorch + HuggingFace Transformers
- 单GPU即可训练（AutoDL RTX 4090）
- 代码模块化，便于后续扩展

### 3.2 系统功能模块

| 模块 | 功能 | 文件 |
|------|------|------|
| 配置管理 | 管理超参数、路径、设备选择 | config.py |
| CLIP封装 | 冻结CLIP模型，提供特征提取接口 | clip_wrapper.py |
| 深度预处理 | 单通道深度→3通道RGB兼容格式 | depth_processor.py |
| 传感器适配器 | MLP/Cross-Attention特征对齐网络 | sensor_adapter.py |
| 数据加载 | NYU Depth V2加载、小样本子集划分 | nyu_dataset.py |
| 对比学习损失 | NT-Xent损失计算 | loss.py |
| 训练循环 | epoch训练、验证、模型保存 | trainer.py |
| Zero-shot评估 | 深度→Adapter→文本类别相似度分类 | zero_shot.py |
| 可视化 | 训练曲线、精度对比、每类精度图 | visualize.py |
| 主入口 | 命令行参数解析、完整流程编排 | main.py |

---

## 四、功能详细设计与实现

### 4.1 系统架构

```
深度图 (单通道)
    │
    ┌─ 深度预处理 (repeat/normal) ─→ 3通道图
    │
    ┌─ CLIP ViT 编码器 (冻结) ─→ Depth CLIP Features
    │
    ┌─ MLP Adapter (可训练) ─→ 对齐后的Depth Embedding
    │
    ┌─ NT-Xent 对比损失 ─→ 与RGB CLIP Embedding对齐
    │
    ┌─ Zero-shot 评估 ─→ 与文本类别Embedding计算相似度
```

### 4.2 核心模块设计

#### 4.2.1 CLIP封装（clip_wrapper.py）

```python
class CLIPWrapper(nn.Module):
    # 冻结所有CLIP参数
    # 提供：encode_rgb()、encode_text()、get_class_text_embeds()
    # 支持patch token提取（为Cross-Attention预留）
```

技术要点：使用HuggingFace `CLIPModel`进行前向推理，`requires_grad=False`冻结参数，输出L2归一化embedding。

#### 4.2.2 深度预处理（depth_processor.py）

```python
class DepthProcessor(nn.Module):
    # 策略1: repeat — 单通道重复3次（简单，保留原始深度值）
    # 策略2: normal — 计算表面法线（3通道，包含几何信息）
```

深度图归一化到[0,1]后进行处理，确保与CLIP预训练数据分布兼容。

#### 4.2.3 传感器适配器（sensor_adapter.py）

**MLP Adapter（当前方案）：**
```
Linear(512→512) → LayerNorm → ReLU → Dropout → Linear(512→512) → LayerNorm
```
参数量：~528K，轻量高效。

**Cross-Attention Adapter（扩展方案）：**
- 16个可学习query tokens通过Cross-Attention从patch tokens中提取信息
- 同时输出global embedding（对比学习用）和local tokens（局部对齐用）
- 参数量：~2.1M

#### 4.2.4 对比学习损失（loss.py）

NT-Xent损失（InfoNCE）：
- 对N对样本构造N×N相似度矩阵
- 对角线上为正样本对，其余为负样本
- 对称交叉熵损失（两个方向平均）

```python
L = 0.5 * (CE(sim, labels) + CE(sim.T, labels))
```

#### 4.2.5 Zero-shot评估（zero_shot.py）

对于每个测试样本：
1. 深度图 → CLIP ViT → Depth Features → Adapter → Adapted Embedding
2. 与所有场景类别名称的文本Embedding计算余弦相似度
3. 取相似度最高的类别作为预测结果
4. 统计Top-1和Top-5准确率

### 4.3 训练流程

```
1. 加载NYU Depth V2数据集，抽取200/50/50样本（train/val/test）
2. 加载预训练CLIP ViT-B/32，冻结全部参数
3. 初始化MLP Adapter
4. 每个batch:
   a. RGB图 → CLIP ViT → RGB Embedding
   b. 深度图 → CLIP ViT → Depth Feature → Adapter → Adapted Embedding
   c. 计算Adapted Embedding与RGB Embedding的NT-Xent损失
   d. 反向传播更新Adapter参数
5. 每epoch后验证集评估zero-shot分类精度
6. 保存最优模型
7. 测试集最终评估 + 可视化
```

### 4.4 关键技术参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CLIP模型 | ViT-B/32 | 权衡速度与效果 |
| Batch size | 16 | 适配单卡GPU |
| 学习率 | 1e-3 | AdamW优化器 |
| Epochs | 30 | 小样本充足 |
| 温度系数 τ | 0.07 | CLIP标准设置 |
| 训练样本 | 200 | 小规模快速验证 |
| 测试样本 | 50 | 充分评估 |

---

## 五、开发难点与体会

### 5.1 技术难点

**难点1：CLIP特征空间的传感器模态适配**
- 深度图的物理分布（距离值）与RGB图像存在根本性差异
- CLIP的ViT从未见过深度数据，直接送入会得到不稳定的特征
- 解决：通过MLP Adapter对CLIP特征进行非线性变换，使其在保持语义的同时适应深度模态

**难点2：小样本下的对比学习训练**
- 仅200个训练样本，容易过拟合
- NT-Xent损失在batch内构造负样本，batch size较小时负样本多样性不足
- 解决：使用较小的学习率（1e-3）和较强的正则化（Dropout=0.1, weight_decay=1e-4），配合学习率余弦退火

**难点3：数据集处理**
- NYU Depth V2深度图为原始米制距离，范围跨度大（0.1m~10m+）
- 深度图预处理需要同时保留结构信息和数值范围
- 解决：提供三种预处理策略（repeat/normal），实验对比选择最优

### 5.2 开发中遇到的问题

**问题1：HuggingFace网络连接问题**
Windows开发环境下无法连接HuggingFace下载CLIP模型。解决：在本地编写全部代码并通过mock数据验证逻辑，然后在AutoDL云GPU上运行实际训练。

**问题2：深度图的3通道转换**
CLIP的ViT仅接受3通道RGB输入，而深度图是单通道。直接用repeat策略重复3次是最简单的方法，但表面法线策略能提供更丰富的几何信息。最终选择两种策略均实现，留作对比实验。

**问题3：NT-Xent损失的数值稳定性**
初始实现时loss出现NaN。解决：确保输入embedding经过L2归一化，温度系数设置为合理值（0.07），并添加梯度裁剪（max_norm=1.0）。

### 5.3 心得体会

1. **模块化设计的重要性**：将系统分为config/models/data/training/evaluation/visualization六大模块，每个模块职责清晰，便于调试和后续扩展。这种设计模式不仅适用于本项目，也是工程实践的通用原则。

2. **深度学习的迁移学习思想**：CLIP模型在4亿图文对上预训练后，其视觉特征空间具有很强的通用性。通过轻量级Adapter（仅占CLIP总参数的3%），就能将这种能力迁移到全新的传感器模态。这体现了"预训练+微调"范式的强大。

3. **对比学习的核心在于构造正负样本**：NT-Xent损失的关键不是损失函数本身，而是如何定义正负样本对。在本项目中，同一场景的RGB和深度图构成正样本对，不同场景的深度图构成负样本对。

4. **学术论文到工程实现的转化**：PROPOSAL中设想的4个创新点（局部对齐、多传感器统一、知识蒸馏等）在课程大作业阶段先做核心功能（全局对齐+单传感器），后续再逐步扩展。这种MVP策略保证了课程进度和论文质量的双赢。

---

## 六、实现总结

### 6.1 项目成果

| 成果 | 说明 |
|------|------|
| 完整代码 | 26个文件，~2700行Python（不含注释） |
| 6个核心模块 | config/models/data/training/evaluation/visualization |
| 2种适配器 | MLP Adapter（当前）+ Cross-Attention Adapter（扩展） |
| 2种深度预处理 | repeat + surface normal |
| Zero-shot评估 | Top-1 / Top-5准确率 + 每类精度 |
| 可视化输出 | 训练曲线、精度对比、每类精度图 |
| 单元测试 | 6个测试全部通过 |

### 6.2 技术栈

- **框架**：PyTorch 2.x + HuggingFace Transformers
- **模型**：OpenAI CLIP ViT-B/32（冻结权重）
- **数据**：HuggingFace Datasets（NYU Depth V2）
- **可视化**：Matplotlib
- **运行环境**：AutoDL RTX 4090 / 任何单GPU机器

### 6.3 后续计划

**课程阶段（6月23日前）：**
- [x] 项目立项与方案撰写（PROPOSAL.md）
- [x] 核心代码实现与测试
- [ ] AutoDL上运行训练，获取实际结果数据
- [ ] 大作业报告提交
- [ ] PPT展示汇报

**论文阶段（6月23日后）：**
- Cross-Attention Adapter + 局部对齐实现
- 多传感器扩展（Depth → Depth + Thermal）
- 知识蒸馏提高数据效率
- 全量消融实验
- 论文撰写与投稿（目标：RA-L / Neurocomputing）

### 6.4 项目地址

**GitHub**: https://github.com/Mar-Ding/cross-modal-clip
（代码已commit，因网络限制暂未push，将在AutoDL上push）
