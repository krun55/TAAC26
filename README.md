# TAAC26

当前仓库代码对应 GitHub: [krun55/TAAC26](https://github.com/krun55/TAAC26)。

本文档基于 `baseline/` 子目录与当前仓库主体代码的 diff 整理。对比范围主要包括：

- `dataset.py`
- `model.py`
- `train.py`
- `trainer.py`
- `infer.py`
- `utils.py`
- `run.sh`

具体提分结果先留空，后续按实验记录补充。

## 当前默认方案

`run.sh` 当前启用的主配置：

```bash
python3 -u train.py \
  --experiment_name sota_activity_dense_utc8_pair6266_bf16train_fp32infer \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --num_workers 8 \
  --use_time_buckets \
  --use_abs_time_emb \
  --time_zone_offset_hours 8 \
  --use_user_activity_dense \
  --use_pair_62_66 \
  --use_bf16_train \
  --inference_dtype float32
```

提分：从0.811到0.8309

## 特性 Diff

### 1. 绝对时间特征

相比 baseline，当前版本新增了用户侧和序列侧的绝对时间建模。

主要变化：

- 从 `timestamp` 派生用户当前样本的 `hour` 和 `dayofweek`。
- 从各序列域时间戳派生 `seq_*_hour` 和 `seq_*_dayofweek`。
- 支持 `--time_zone_offset_hours`，当前默认脚本使用 UTC+8。
- 支持 `--user_abs_time_missing_as_padding`，可将缺失时间作为 padding 处理。
- 模型侧新增 hour/dayofweek embedding，并注入用户 token 与序列 token。

相关开关：

- `--use_abs_time_emb`
- `--time_zone_offset_hours`
- `--user_abs_time_missing_as_padding`
- `--add_user_time_to_dense_tok`

提分：0.012

### 2. 用户活跃度 Dense 特征

当前版本从用户历史序列中构造额外 dense 特征，用于表达短期活跃度与历史长度。

主要变化：

- 新增 synthetic dense fid `900-915`。
- 对 `seq_a`、`seq_b`、`seq_c`、`seq_d` 记录序列长度。
- 统计 1 天、3 天、5 天窗口内的历史行为次数。
- 使用 `log1p` 压缩长度和计数特征。
- 数据集层自动追加到 `user_dense_feats`，模型侧自动识别并分支处理。

相关开关：

- `--use_user_activity_dense`
- `--no_user_activity_dense`

提分：0.0005

### 3. FID 62-66 稀疏/Dense 配对建模

当前版本对 fid `62-66` 做了专门处理，将同一 fid 的 sparse id 与 dense 值配对进入 token 化流程。

主要变化：

- 新增 `PAIR_62_66_FIDS = (62, 63, 64, 65, 66)`。
- 在 `GroupNSTokenizer` 与 `RankMixerNSTokenizer` 中支持 pair dense 输入。
- 对 fid `62-66` 校验 sparse/dense schema 必须同时存在且长度匹配。
- 将配对特征从普通 dense 分支中拆出，避免重复建模。

相关开关：

- `--use_pair_62_66`

提分：0.06

### 4. RankMixer NS Tokenizer 默认化

baseline 已有 RankMixer 入口，当前版本将其作为默认运行方案继续强化，并配合新特征使用。

主要变化：

- 默认使用 `--ns_tokenizer_type rankmixer`。
- 默认使用 `--user_ns_tokens 5`、`--item_ns_tokens 2`。
- 默认 `--num_queries 2`。
- 支持与 `use_pair_62_66` 配合，将配对特征注入 NS token。
- 支持 `randomized_split`，可按 seed 对 RankMixer 的 fid 分组做随机切分实验。

相关开关：

- `--randomized_split`
- `--randomized_split_seed`

提分：_=负收益

### 5. Dense Cross Token

当前版本新增可选 dense cross token，用 selected dense 特征显式构造用户与目标侧交互 token。

主要变化：

- 支持从用户 dense 特征选择 user 侧输入。
- 支持目标侧来源为 `user_dense`、`item_dense` 或 `item_ns`。
- 通过投影、LayerNorm、交互后再映射为一个语义 token。
- 该 token 可追加进 NS token 集合参与后续混合。

相关开关：

- `--use_dense_cross_token`
- `--dense_cross_user_fids`
- `--dense_cross_item_source`
- `--dense_cross_item_fids`
- `--dense_cross_dim`

提分：掉大分

### 6. NS Token 交互增强

当前版本在 NS token 层增加多种可选交互模块。

主要变化：

- 新增 `NSTokenSEGate`，对 NS tokens 做 squeeze-and-excitation 风格门控。
- 新增 `NSSelfAttention`，支持 NS token 间自注意力。
- 新增 `NSAutoIntStack`，用 gated AutoInt interacting layers 建模 NS token 交互。
- 新增 `GatedTokenAutoInt`，可在 multi-seq mixer 前后对组合 token 做 AutoInt。

相关开关：

- `--use_ns_se_gate`
- `--use_ns_self_attn`
- `--ns_self_attn_impl`
- `--ns_autoint_layers`
- `--ns_autoint_heads`
- `--ns_autoint_gate_init`
- `--ns_autoint_activation`
- `--use_combined_autoint`
- `--combined_autoint_layers`
- `--combined_autoint_heads`
- `--combined_autoint_dropout`
- `--combined_autoint_gate_init`
- `--combined_autoint_position`

提分：NSTokenSEGate和NSSelfAttention一起0.002

### 7. Target-Aware Attention

当前版本新增 DIN 风格目标感知序列读出，以 item NS token 作为 target 表征，对多序列 token 做目标相关聚合。

主要变化：

- 新增 `TargetAwareAttentionHead`。
- 用 target 表征、序列均值、最后有效 token 和 DIN context 组合生成残差增强。
- 作为 head 表征的 gated residual，降低直接替换主干带来的风险。

相关开关：

- `--use_target_attention`

提分：负收益

### 8. Head Cross Network

当前版本在最终 head 表征上加入可选 residual cross layers，用于补充高阶交叉。

主要变化：

- 新增 `HeadCrossNet`。
- 支持配置层数与初始化尺度。
- 仅在 `head_cross_layers > 0` 时启用。

相关开关：

- `--head_cross_layers`
- `--head_cross_init_scale`

提分：负收益

### 9. Item Dense 支持

baseline 中 `item_dense_feats` 固定为空；当前版本支持从 schema 和 parquet 中读取 item dense 特征。

主要变化：

- 解析 schema 中的 `item_dense` 配置。
- 构建 `item_dense_schema` 与 `_item_dense_plan`。
- batch 输出真实 `item_dense_feats`。
- 可供 dense cross token 等模块使用。

提分：负收益

### 10. 训练稳定性与效率

当前版本新增 bf16 训练、EMA、epoch checkpoint 等训练侧改动。

主要变化：

- 支持训练阶段 `torch.autocast` bf16。
- inference 仍强制使用 float32，降低提交侧数值风险。
- 新增 `ModelEMA`，可从指定 epoch 开始维护滑动平均权重。
- validation 和 checkpoint 保存时临时切换到 EMA shadow weights。
- checkpoint 命名改为 `epochN.global_stepM...`。
- 每个 epoch 保存 checkpoint，并写入 `schema.json`、`ns_groups.json`、`train_config.json`。
- 稀疏参数重初始化后同步 EMA shadow，避免 EMA 引用旧参数状态。

相关开关：

- `--use_bf16_train`
- `--inference_dtype`
- `--ema_decay`
- `--ema_start_epoch`

提分：ema实现错了，收益很小，万分位

### 11. 推理一致性与配置校验

当前版本强化了训练-推理一致性，减少 checkpoint 与提交代码不匹配时的隐式 fallback。

主要变化：

- `infer.py` 必须读取 checkpoint 目录中的 `train_config.json`。
- 对关键结构参数做 required key 校验。
- 校验 `use_time_buckets` 与 `num_time_buckets` 是否一致。
- 按训练配置重建绝对时间、活跃度 dense、pair 62-66、dense cross、NS 交互等结构。
- 支持旧 checkpoint 中 `ns_self_attn` 到 `ns_feature_cross` 的 key remap。
- 推理 dtype 目前限制为 `float32`。

提分：不计入



## Diff 复现方式

本地可用下面的方式复现 baseline 对比：

```bash
for f in dataset.py infer.py model.py ns_groups.json run.sh train.py trainer.py utils.py; do
  diff -u "baseline/$f" "$f"
done
```
