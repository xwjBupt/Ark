# 仅使用 ImageNet 初始化的胸片多标签实验

这组实验对应用户反馈：此前医学 Ark 权重版本的线上分数为 79.67，现要求重新训练不使用胸片预训练权重的模型。旧模型和旧提交文件保留，新产物存放于 `artifacts/cxr_imagenet/`。

## 初始化与数据

- 骨干：Swin-Base，微软公开的 ImageNet-22K 预训练、ImageNet-1K 微调权重。
- 权重：`pretrained_ckpts/imagenet/swin_base_patch4_window7_224_22kto1k.pth`。
- SHA-256：`a5348473e69277d69db915362bef760b5fd04362057bd59c7fe064d431532eab`。
- 来源：<https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth>。
- 上游说明：<https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md>。
- 许可证：MIT；完整原文复制到 `artifacts/cxr_imagenet/licenses/swin_base_LICENSE.txt`。
- 未使用 Ark 医学预训练权重、医学教师、旧预测软标签或外部医学数据；新分类头和融合头随机初始化。
- 仅官方训练集参与参数优化；官方验证集用于早停、模型/融合权重和每类阈值选择；模型和阈值冻结后才对测试集推理。
- 训练、验证标签逐条与 CSV 对照；检查结果保存在 `label_audit.json`。

## 四组实验

| GPU | 名称 | 分辨率 | 随机种子 | 骨干学习率 |
|---|---|---|---|---|
| 4 | imagenet_swin_384_s42 | 384 | 42 | 3e-5 |
| 5 | imagenet_swin_448_s2026 | 448 | 2026 | 2e-5 |
| 6 | imagenet_swin_384_s3407 | 384 | 3407 | 3e-5 |
| 7 | imagenet_swin_448_s17 | 448 | 17 | 2e-5 |

共享骨干提取每个视角的特征，按疾病独立学习视角注意力，输出 10 个独立 sigmoid 概率。注意力归一化的维度是图像/视角，疾病类别之间没有 softmax。训练标签采用长度为 10 的 multi-hot 向量，并用类别加权 BCE 监督。训练最多抽取每次检查两张互补视角；验证和最终推理使用该检查所有图像。

最多 30 轮，两轮学习率预热，至少训练 12 轮，连续 8 轮验证 Macro-AUC 不提升时早停；保存最佳 EMA 权重。使用 BF16 训练、FP16 autocast 推理。显存由实际前后向和优化器探测确定，384/448 的图像批次预算分别约为 88/68，实际显存约 41–43GB/卡。

## 查看、恢复与导出

已运行的 supervisor PID 见 `artifacts/cxr_imagenet/supervisor.pid`，请勿重复启动。

```bash
/home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/monitor.py \
  --output artifacts/cxr_imagenet
tail -f artifacts/cxr_imagenet/runs/imagenet_swin_448_s2026/train.log
```

中断后，先确认 GPU 4–7 空闲、旧 supervisor 与其训练子进程均已退出，再恢复：

```bash
/home/wjx/miniconda3/envs/nnunet_seg/bin/python experiments/cxr_multiview/run_imagenet.py --resume
```

四组训练结束后自动用导出的最佳权重重新评估验证集。在总权重不超过 500,000,000 字节的候选中，以验证 Macro-AUC 最高者为最终模型；比较单模型及权重为 0.25/0.5/0.75 的两模型组合。导出前再次核对 ImageNet 初始化来源和 SHA-256。

最终目录 `artifacts/cxr_imagenet/final/`：

- `submission.csv`：`Subject_id,Study_id,Predict_class,Probability,Gate` 五列，每个 Study_id 含 1–10 类，共 14,140 行。
- `bundle.json`：冻结后的模型、融合权重、每类阈值、验证指标、初始化来源及模型哈希。
- `model_comparison.json`：验证集候选比较。
- `validation_predictions.npz`：最终验证集概率。
- 一个或两个推理权重文件，不包含医学 Ark 权重。

测试集没有标签，本地验证指标不能当作线上得分。T4 的端到端推理延迟仍需按赛事硬件复测；ImageNet 来源核验不等于已满足全部部署限制。
