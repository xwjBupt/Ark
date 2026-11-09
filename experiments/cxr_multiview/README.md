# 胸片多视角、多标签实验

本目录在 Ark 工程中提供独立训练入口，不依赖原工程的单图数据加载器。用户明确要求继续使用本地医学 Ark 权重。本组实验因此不符合保存的《参赛规则》第 1 条及补充说明第 2、3 节；权重来源与这一限制会写入实验和最终导出记录。不得将本组结果申报为非医学预训练方案。

## 当前实验

| 物理 GPU | 实验 | 初始化 | 分辨率 | 种子 |
|---|---|---|---|---|
| 4 | ark6_media_384_s42 | ark6_swinbase_224_ep200.pth.tar 的 teacher | 384 | 42 |
| 5 | ark6_media_448_s2026 | 同上 | 448 | 2026 |
| 6 | ark5_media_384_s3407 | ark5_swinbase_224_ep200.pth.tar 的 teacher | 384 | 3407 |
| 7 | ark6_miccai_448_s17 | ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar | 448 | 17 |

运行环境：`/home/wjx/miniconda3/envs/nnunet_seg/bin/python`。仅使用其中现有依赖，没有修改该环境，也没有触碰 GPU 0–3 的任务。基础 Python 环境的 cuDNN 初始化失败，因此没有用于训练。

实验产物统一保存在 `/home/wjx/CodeData/code/Ark/artifacts/cxr_multiview/`。后台 supervisor 与各训练进程独立于终端运行，PID 和状态保存在 `supervisor.pid`、`status.json` 中。

在 Ark 根目录查看状态：

```bash
/home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/monitor.py
nvidia-smi
tail -f artifacts/cxr_multiview/runs/ark6_media_384_s42/train.log
```

每组最多训练 30 轮，最少 12 轮，验证集 AUC 连续 8 轮不提升时早停。每轮保存完整续训状态，每次 AUC 提升保存 FP16 最佳权重。四组结束后自动执行 `finalize.py`，从导出的实际权重重新计算验证预测，选择模型并预测测试集。首次完整训练需要数小时，以实际日志速度为准。

## 数据与模型

- 原始数据：`/home/wjx/CodeData/data/cxr/cxr_first`。
- 按 `Study_id` 分组；标签以分号拆分，严格映射类别 1–10。每次检查只计一个训练样本。
- 训练集 18,513 次检查、30,344 张图；验证集 859 次检查、1,422 张图；测试集 1,414 次检查、2,218 张图。三个集合的患者不重叠。
- 数据并非固定两张。训练时最多抽取两张，优先同时选择正位和侧位；验证和测试使用该检查全部图像。单视角检查直接参与训练，不复制、不伪造侧位。
- 原图转灰度，保持长宽比缩放并填充至 512×512，缓存为无损 PNG；再缩放到训练分辨率。原图推理与缓存预处理一致。不计算测试集统计量。
- Swin-Base 共享编码器 + 视角 embedding + 每个类别独立的视角注意力。医学预训练的 projector、原任务分类头不迁移，编码器参数完整迁移；加载缺失或不匹配会报错。
- 损失是温和类别加权 BCE，加 0.15 倍图像级辅助 BCE；类别权重仅由训练集计算。支持同一检查多种疾病，使用 sigmoid，不使用 softmax。
- AdamW，骨干学习率 2e-5/3e-5，分类部分 3e-4，两轮 warmup 后 cosine 衰减，梯度裁剪、EMA、BF16 训练和 FP16 推理。训练增强为小幅仿射和亮度/对比度扰动。
- 显存调优执行真实的前向、反向、AdamW 更新，按图像数组织批次。当前 384 输入每批 88 张图，448 输入每批 68 张；每批检查数量随单/多视角比例变化。没有通过无效张量占用显存。

## 指标、选择与产物

保存的官方说明以 **Macro-AUC** 为主、**Macro-F1** 为同分辅助指标；同时记录 Macro-AP 和梯形积分 PR-AUC，二者不混称。所有模型选择和 F1 阈值拟合仅使用验证集。验证集不并入训练；测试集不用于训练、伪标签、归一化、阈值或模型选择。

`runs/<实验名>/`：

- `config.json`：初始化、分辨率、种子、学习率、GPU 等。
- `autotune.json`：显存探测和批次选择结果。
- `train.log`、`history.jsonl`：训练过程、每轮 AUC/AP/PR-AUC/F1。
- `best.pt`、`best_metrics.json`：最佳 EMA 模型，约 174 MB。
- `last.pt`：完整优化器、模型、EMA、随机状态，供续训；不属于推理权重。
- `DONE.json`：该实验完成后才会生成。

最终输出 `final/`：

- `submission.csv`：每个 Study_id 对应 10 行，共 14,140 行，列为 `Subject_id,Study_id,Predict_class,Probability,Gate`，UTF-8 编码。
- `bundle.json`：固定模型、融合权重、每类阈值、哈希、验证指标与初始化限制说明。
- `model_comparison.json`：单模型及两模型融合的验证结果。
- `validation_predictions.npz`：最终验证预测。
- 仅推理使用的 FP16 权重。

仅考虑总推理权重与配置不超过 500,000,000 字节的单模型/两模型组合；两模型权重只搜索 0.25/0.5/0.75，并要求较最佳单模型提升至少 0.001 AUC。最终推理使用固定前向和阈值，关闭随机增强，不进行测试时拟合。

500MB 限制已在导出中检查；指定 T4 的 100ms 延迟尚未实测，不能用 L40 训练速度证明达标。最终融合可能需要单模型或额外推理优化才能满足该时间约束。医学初始化这一规则冲突不会由模型压缩或加速消除。

完成后重新生成预测（使用原图确定性预处理）：

```bash
CUDA_VISIBLE_DEVICES=4 /home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/predict.py \
  --bundle artifacts/cxr_multiview/final/bundle.json \
  --manifest artifacts/cxr_multiview/data/test.json \
  --output artifacts/cxr_multiview/final/submission.csv
```

原始数据路径记录在 manifest 中。迁移数据目录时须重新执行 `prepare.py`，或更新 manifest 的 `raw` 路径。

## 准备、恢复和检查

```bash
# 首次准备；不会更改原始数据
/home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/prepare.py \
  --data-root /home/wjx/CodeData/data/cxr/cxr_first \
  --output /home/wjx/CodeData/code/Ark/artifacts/cxr_multiview/data --workers 24

# 当前已有 supervisor 运行，不要重复启动。
# 确认旧任务退出且 GPU 空闲后，恢复未完成实验：
/home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/run_experiments.py --resume

# 快速检查数据、无跨检查混合、缓存一致性、指标和提交格式
CUDA_VISIBLE_DEVICES='' /home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/test_pipeline.py
```

`smoke/` 仅用于短流程验证，其模拟测试集来自验证集子集，不是比赛测试集；其指标不代表正式实验成绩。

`fetch_weights.py` 是此前准备的 ImageNet 下载工具，其下载结果未用于当前 Ark 实验。未解压本地不完整的 `Ark+_Nature.zip`。
