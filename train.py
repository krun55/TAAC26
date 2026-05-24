"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 6] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PAIR_62_66_FIDS, PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_dense_feature_specs(schema: FeatureSchema) -> List[Tuple[int, int, int]]:
    """Build dense specs of the form ``[(fid, offset, length), ...]``."""
    return [(fid, offset, length) for fid, offset, length in schema.entries]


def build_pair_feature_specs(
    user_int_schema: FeatureSchema,
    user_dense_schema: FeatureSchema,
) -> List[Tuple[int, int, int, int, int]]:
    """Build pair specs for fids 62-66: (fid, int_offset, int_len, dense_offset, dense_len)."""
    specs: List[Tuple[int, int, int, int, int]] = []
    for fid in PAIR_62_66_FIDS:
        try:
            int_offset, int_len = user_int_schema.get_offset_length(fid)
            dense_offset, dense_len = user_dense_schema.get_offset_length(fid)
        except KeyError as exc:
            raise KeyError(
                f"use_pair_62_66=True requires fid {fid} in both user_int and user_dense schema"
            ) from exc
        if int_len != dense_len:
            raise ValueError(
                f"use_pair_62_66=True requires fid {fid} int length ({int_len}) "
                f"to equal dense length ({dense_len})"
            )
        specs.append((fid, int_offset, int_len, dense_offset, dense_len))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=6,
                        help='Fixed number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Deprecated; early stopping is disabled')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')
    parser.add_argument('--use_user_activity_dense',
                        dest='use_user_activity_dense',
                        action='store_true',
                        default=True,
                        help='Enable synthetic user activity dense features '
                             '(default on).')
    parser.add_argument('--no_user_activity_dense',
                        dest='use_user_activity_dense',
                        action='store_false',
                        help='Disable synthetic user activity dense features.')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')
    parser.add_argument('--use_abs_time_emb', action='store_true', default=False,
                        help='Enable absolute hour/dayofweek embeddings '
                             '(default off; user and sequence tables are separate).')
    parser.add_argument('--time_zone_offset_hours', type=int, default=0,
                        help='Hour offset applied before deriving absolute hour/dayofweek '
                             '(0 = UTC, 8 = GMT+8).')
    parser.add_argument('--user_abs_time_missing_as_padding', action='store_true',
                        default=False,
                        help='Reserve user absolute time id 0 for timestamp<=0 '
                             'and shift valid user hour/dayofweek ids by +1.')
    parser.add_argument('--add_user_time_to_dense_tok', action='store_true', default=False,
                        help='Also add user absolute time embedding to the user dense token '
                             '(default off).')
    parser.add_argument('--use_pair_62_66', action='store_true', default=False,
                        help='Enable weighted mean pooling for user pair fids 62-66 '
                             '(default off).')
    parser.add_argument('--use_dense_cross_token', action='store_true', default=False,
                        help='Append a RankUp-style dense cross token to the NS token '
                             'sequence (default off).')
    parser.add_argument('--dense_cross_user_fids', type=str, default='61,87',
                        help='Comma-separated user_dense fids used as the user side '
                             'of the dense cross token.')
    parser.add_argument('--dense_cross_item_source', type=str, default='user_dense',
                        choices=['user_dense', 'item_dense', 'item_ns'],
                        help='Source for the item side of the dense cross token.')
    parser.add_argument('--dense_cross_item_fids', type=str, default='89,90,91',
                        help='Comma-separated fids for dense_cross_item_source. '
                             'Ignored when dense_cross_item_source=item_ns.')
    parser.add_argument('--dense_cross_dim', type=int, default=0,
                        help='Hidden dimension for aligned dense cross projections '
                             '(0 = d_model).')
    parser.add_argument('--use_target_attention', action='store_true', default=False,
                        help='Enable DIN-style target-aware sequence readout '
                             '(default off).')
    parser.add_argument('--use_ns_se_gate', action='store_true', default=False,
                        help='Enable SE-style gating over NS tokens '
                             '(default off).')
    parser.add_argument('--use_ns_self_attn', action='store_true', default=False,
                        help='Enable position-neutral self-attention among NS '
                             'tokens inside each HyFormerBlock (default off).')
    parser.add_argument('--ns_self_attn_impl', type=str, default='mha',
                        choices=['mha', 'autoint'],
                        help='Implementation used when --use_ns_self_attn is enabled: '
                             'legacy MHA or AutoInt stack.')
    parser.add_argument('--ns_autoint_layers', type=int, default=2,
                        help='Number of AutoInt interacting layers for NS tokens.')
    parser.add_argument('--ns_autoint_heads', type=int, default=0,
                        help='Number of attention heads for NS AutoInt; '
                             '0 means reuse --num_heads.')
    parser.add_argument('--ns_autoint_gate_init', type=float, default=0.05,
                        help='Initial residual interpolation gate for NS AutoInt.')
    parser.add_argument('--ns_autoint_activation', type=str, default='relu',
                        choices=['relu', 'silu', 'gelu'],
                        help='Activation used inside NS AutoInt interacting layer.')
    parser.add_argument('--use_combined_autoint', action='store_true', default=False,
                        help='Enable gated AutoInt residual over combined decoded-Q and NS tokens.')
    parser.add_argument('--combined_autoint_layers', type=int, default=1,
                        help='Number of AutoInt layers over combined tokens when enabled.')
    parser.add_argument('--combined_autoint_heads', type=int, default=0,
                        help='AutoInt heads over combined tokens; 0 means inherit --num_heads.')
    parser.add_argument('--combined_autoint_dropout', type=float, default=-1.0,
                        help='AutoInt dropout; negative means inherit --dropout_rate.')
    parser.add_argument('--combined_autoint_gate_init', type=float, default=0.0,
                        help='Initial scalar residual gate for combined AutoInt.')
    parser.add_argument('--combined_autoint_position', type=str, default='pre_mixer',
                        choices=['pre_mixer', 'post_mixer'],
                        help='Where to apply combined AutoInt relative to RankMixer.')
    parser.add_argument('--head_cross_layers', type=int, default=0,
                        help='Number of final-head CrossNet residual layers '
                             '(0 = disabled).')
    parser.add_argument('--head_cross_init_scale', type=float, default=0.01,
                        help='Initial learnable scale for final-head CrossNet '
                             'residual layers.')
    parser.add_argument('--use_bf16_train', action='store_true', default=False,
                        help='Enable bfloat16 autocast for training forward/loss only '
                             '(default off).')
    parser.add_argument('--inference_dtype', type=str, default='float32',
                        choices=['float32'],
                        help='Inference dtype recorded for infer.py; only float32 is supported.')
    parser.add_argument('--experiment_name', type=str, default='baseline',
                        help='Free-form experiment name recorded in train_config.json. '
                             'Use baseline_abs_time_pair6266_bf16train_fp32infer '
                             'for the minimal new experiment.')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay rate for model weights '
                             '(0 = disabled; typical value: 0.999)')
    parser.add_argument('--ema_start_epoch', type=int, default=3,
                        help='Start EMA from this 1-based epoch '
                             '(default: 3; only effective when --ema_decay > 0)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--randomized_split', action='store_true', default=False,
                        help='Enable RankUp randomized permutation splitting '
                             'for RankMixer sparse NS tokens')
    parser.add_argument('--randomized_split_seed', type=int, default=42,
                        help='Seed for randomized permutation splitting')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")
    if args.ns_self_attn_impl == 'autoint' and not args.use_ns_self_attn:
        logging.warning(
            "ns_self_attn_impl=autoint only takes effect when "
            "--use_ns_self_attn is enabled.")
    if args.ns_autoint_layers < 1:
        raise ValueError("ns_autoint_layers must be >= 1")
    resolved_ns_autoint_heads = (
        args.ns_autoint_heads if args.ns_autoint_heads > 0 else args.num_heads
    )
    if args.ns_self_attn_impl == 'autoint':
        if resolved_ns_autoint_heads < 1:
            raise ValueError("ns_autoint_heads must resolve to >= 1")
        if args.d_model % resolved_ns_autoint_heads != 0:
            raise ValueError(
                f"d_model={args.d_model} must be divisible by NS AutoInt "
                f"heads={resolved_ns_autoint_heads}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        time_zone_offset_hours=args.time_zone_offset_hours,
        user_abs_time_missing_as_padding=args.user_abs_time_missing_as_padding,
        use_user_activity_dense=args.use_user_activity_dense,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "user_dense_feature_specs": build_dense_feature_specs(
            pcvr_dataset.user_dense_schema),
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "item_dense_feature_specs": build_dense_feature_specs(
            pcvr_dataset.item_dense_schema),
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "randomized_split": args.randomized_split,
        "randomized_split_seed": args.randomized_split_seed,
        "use_abs_time_emb": args.use_abs_time_emb,
        "user_abs_time_missing_as_padding": args.user_abs_time_missing_as_padding,
        "add_user_time_to_dense_tok": args.add_user_time_to_dense_tok,
        "use_pair_62_66": args.use_pair_62_66,
        "use_dense_cross_token": args.use_dense_cross_token,
        "dense_cross_user_fids": args.dense_cross_user_fids,
        "dense_cross_item_source": args.dense_cross_item_source,
        "dense_cross_item_fids": args.dense_cross_item_fids,
        "dense_cross_dim": args.dense_cross_dim,
        "use_target_attention": args.use_target_attention,
        "use_ns_se_gate": args.use_ns_se_gate,
        "use_ns_self_attn": args.use_ns_self_attn,
        "ns_self_attn_impl": args.ns_self_attn_impl,
        "ns_autoint_layers": args.ns_autoint_layers,
        "ns_autoint_heads": args.ns_autoint_heads,
        "ns_autoint_gate_init": args.ns_autoint_gate_init,
        "ns_autoint_activation": args.ns_autoint_activation,
        "use_combined_autoint": args.use_combined_autoint,
        "combined_autoint_layers": args.combined_autoint_layers,
        "combined_autoint_heads": args.combined_autoint_heads,
        "combined_autoint_dropout": args.combined_autoint_dropout,
        "combined_autoint_gate_init": args.combined_autoint_gate_init,
        "combined_autoint_position": args.combined_autoint_position,
        "head_cross_layers": args.head_cross_layers,
        "head_cross_init_scale": args.head_cross_init_scale,
    }
    if args.use_pair_62_66:
        model_args["user_pair_feature_specs"] = build_pair_feature_specs(
            pcvr_dataset.user_int_schema,
            pcvr_dataset.user_dense_schema,
        )

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    train_config = dict(vars(args))
    train_config["num_time_buckets"] = model_args["num_time_buckets"]

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=train_config,
        use_bf16_train=args.use_bf16_train,
        ema_decay=args.ema_decay,
        ema_start_epoch=args.ema_start_epoch,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
