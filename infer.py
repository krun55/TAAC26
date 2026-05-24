"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved from the ckpt directory's ``train_config.json``
(written by ``trainer.py`` when saving a checkpoint). Missing config files or
required keys fail fast so inference cannot silently use the wrong structure.

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``epochN.global_stepM``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PAIR_62_66_FIDS, PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = [
    'd_model',
    'emb_dim',
    'num_queries',
    'num_hyformer_blocks',
    'num_heads',
    'seq_encoder_type',
    'hidden_mult',
    'dropout_rate',
    'seq_top_k',
    'seq_causal',
    'action_num',
    'num_time_buckets',
    'rank_mixer_mode',
    'use_rope',
    'rope_base',
    'emb_skip_threshold',
    'seq_id_threshold',
    'ns_tokenizer_type',
    'user_ns_tokens',
    'item_ns_tokens',
    'randomized_split',
    'randomized_split_seed',
    'use_abs_time_emb',
    'user_abs_time_missing_as_padding',
    'add_user_time_to_dense_tok',
    'use_pair_62_66',
    'use_dense_cross_token',
    'dense_cross_user_fids',
    'dense_cross_item_source',
    'dense_cross_item_fids',
    'dense_cross_dim',
    'use_target_attention',
    'use_ns_se_gate',
    'use_ns_self_attn',
    'use_combined_autoint',
    'combined_autoint_layers',
    'combined_autoint_heads',
    'combined_autoint_dropout',
    'combined_autoint_gate_init',
    'combined_autoint_position',
    'head_cross_layers',
    'head_cross_init_scale',
]

_MODEL_CFG_DEFAULTS = {
    'use_target_attention': False,
    'use_ns_se_gate': False,
    'use_ns_self_attn': False,
    'use_combined_autoint': False,
    'combined_autoint_layers': 1,
    'combined_autoint_heads': 0,
    'combined_autoint_dropout': -1.0,
    'combined_autoint_gate_init': 0.0,
    'combined_autoint_position': 'pre_mixer',
    'head_cross_layers': 0,
    'head_cross_init_scale': 0.01,
    'user_abs_time_missing_as_padding': False,
    'randomized_split': False,
    'randomized_split_seed': 42,
    'use_dense_cross_token': False,
    'dense_cross_user_fids': '61,87',
    'dense_cross_item_source': 'user_dense',
    'dense_cross_item_fids': '89,90,91',
    'dense_cross_dim': 0,
}

_REQUIRED_TRAIN_CONFIG_KEYS = [
    'use_time_buckets',
    'use_abs_time_emb',
    'time_zone_offset_hours',
    'add_user_time_to_dense_tok',
    'use_pair_62_66',
    'use_bf16_train',
    'inference_dtype',
    'num_time_buckets',
]


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def build_dense_feature_specs(schema: FeatureSchema) -> List[Tuple[int, int, int]]:
    return [(fid, offset, length) for fid, offset, length in schema.entries]


def build_pair_feature_specs(
    user_int_schema: FeatureSchema,
    user_dense_schema: FeatureSchema,
) -> List[Tuple[int, int, int, int, int]]:
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


def require_train_config_key(train_config: Dict[str, Any], key: str) -> Any:
    if key not in train_config:
        raise KeyError(f"train_config.json missing required key: {key}")
    return train_config[key]


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    raise FileNotFoundError(
        f"train_config.json not found in {model_dir}. "
        "Inference requires the training config and does not use fallback defaults.")


def validate_train_config(train_config: Dict[str, Any]) -> None:
    for key in _REQUIRED_TRAIN_CONFIG_KEYS:
        require_train_config_key(train_config, key)


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``.

    Newly added structural keys may use defaults so older checkpoints keep
    rebuilding as their original no-head-cross model.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key in train_config:
            cfg[key] = train_config[key]
        elif key in _MODEL_CFG_DEFAULTS:
            cfg[key] = _MODEL_CFG_DEFAULTS[key]
        else:
            cfg[key] = require_train_config_key(train_config, key)
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)
    model_kwargs = dict(model_cfg)
    if model_kwargs.get('use_pair_62_66', False):
        model_kwargs['user_pair_feature_specs'] = build_pair_feature_specs(
            dataset.user_int_schema,
            dataset.user_dense_schema,
        )

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        user_dense_feature_specs=build_dense_feature_specs(
            dataset.user_dense_schema),
        item_dense_dim=dataset.item_dense_schema.total_dim,
        item_dense_feature_specs=build_dense_feature_specs(
            dataset.item_dense_schema),
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_kwargs,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = _remap_legacy_ns_self_attn_keys(model, state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def _remap_legacy_ns_self_attn_keys(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Map n220/n225 checkpoint keys after the inference-side module rename."""
    model_keys = set(model.state_dict().keys())
    old = '.ns_self_attn.'
    new = '.ns_feature_cross.'
    remapped: Dict[str, torch.Tensor] = {}
    changed = 0
    for key, value in state_dict.items():
        new_key = key.replace(old, new)
        if new_key != key and key not in model_keys and new_key in model_keys:
            remapped[new_key] = value
            changed += 1
        else:
            remapped[key] = value
    if changed:
        logging.info(
            "Remapped %d legacy checkpoint keys from ns_self_attn to "
            "ns_feature_cross.",
            changed,
        )
    return remapped


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_hours: Dict[str, torch.Tensor] = {}
    seq_dayofweeks: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))
        seq_hours[domain] = device_batch.get(
            f'{domain}_hour',
            torch.zeros(B, L, dtype=torch.long, device=device))
        seq_dayofweeks[domain] = device_batch.get(
            f'{domain}_dayofweek',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        user_hour=device_batch.get('hour'),
        user_dayofweek=device_batch.get('dayofweek'),
        seq_hours=seq_hours,
        seq_dayofweeks=seq_dayofweeks,
    )


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)
    validate_train_config(train_config)
    inference_dtype = require_train_config_key(train_config, 'inference_dtype')
    if inference_dtype != 'float32':
        raise ValueError(
            f"Only float32 inference is supported, got inference_dtype={inference_dtype!r}")
    use_time_buckets = require_train_config_key(train_config, 'use_time_buckets')
    expected_num_time_buckets = NUM_TIME_BUCKETS if use_time_buckets else 0
    actual_num_time_buckets = require_train_config_key(train_config, 'num_time_buckets')
    if actual_num_time_buckets != expected_num_time_buckets:
        raise ValueError(
            "train_config.json has inconsistent time-bucket config: "
            f"use_time_buckets={use_time_buckets} implies num_time_buckets="
            f"{expected_num_time_buckets}, got {actual_num_time_buckets}")
    use_user_activity_dense = bool(
        train_config.get('use_user_activity_dense', False))
    logging.info(
        "Restored feature switches from train_config: "
        f"use_time_buckets={use_time_buckets}, "
        f"use_user_activity_dense={use_user_activity_dense}, "
        f"use_abs_time_emb={require_train_config_key(train_config, 'use_abs_time_emb')}, "
        f"time_zone_offset_hours={require_train_config_key(train_config, 'time_zone_offset_hours')}, "
        f"user_abs_time_missing_as_padding={train_config.get('user_abs_time_missing_as_padding', False)}, "
        f"add_user_time_to_dense_tok={require_train_config_key(train_config, 'add_user_time_to_dense_tok')}, "
        f"use_pair_62_66={require_train_config_key(train_config, 'use_pair_62_66')}, "
        f"use_dense_cross_token={train_config.get('use_dense_cross_token', False)}, "
        f"dense_cross_user_fids={train_config.get('dense_cross_user_fids', '61,87')}, "
        f"dense_cross_item_source={train_config.get('dense_cross_item_source', 'user_dense')}, "
        f"dense_cross_item_fids={train_config.get('dense_cross_item_fids', '89,90,91')}, "
        f"dense_cross_dim={train_config.get('dense_cross_dim', 0)}, "
        f"use_target_attention={train_config.get('use_target_attention', False)}, "
        f"use_ns_se_gate={train_config.get('use_ns_se_gate', False)}, "
        f"use_ns_self_attn={train_config.get('use_ns_self_attn', False)}, "
        f"use_combined_autoint={train_config.get('use_combined_autoint', False)}, "
        f"combined_autoint_layers={train_config.get('combined_autoint_layers', 1)}, "
        f"combined_autoint_heads={train_config.get('combined_autoint_heads', 0)}, "
        f"combined_autoint_dropout={train_config.get('combined_autoint_dropout', -1.0)}, "
        f"combined_autoint_gate_init={train_config.get('combined_autoint_gate_init', 0.0)}, "
        f"combined_autoint_position={train_config.get('combined_autoint_position', 'pre_mixer')}, "
        f"randomized_split={train_config.get('randomized_split', False)}, "
        f"randomized_split_seed={train_config.get('randomized_split_seed', 42)}, "
        f"use_bf16_train={require_train_config_key(train_config, 'use_bf16_train')}, "
        f"inference_dtype={inference_dtype}")

    # ---- Parse seq_max_lens ----
    sml_str = require_train_config_key(train_config, 'seq_max_lens')
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(require_train_config_key(train_config, 'batch_size'))
    num_workers = int(require_train_config_key(train_config, 'num_workers'))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        time_zone_offset_hours=int(require_train_config_key(
            train_config, 'time_zone_offset_hours')),
        user_abs_time_missing_as_padding=bool(
            train_config.get('user_abs_time_missing_as_padding', False)),
        use_user_activity_dense=use_user_activity_dense,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means MODEL_OUTPUT_PATH is not pointed at a "
            "single epoch checkpoint directory, or that checkpoint is "
            "incomplete and does not contain model.pt."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.float()
    model.eval()
    logging.info(
        f"Model loaded successfully for float32 inference; "
        f"first parameter dtype={next(model.parameters()).dtype}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **({'prefetch_factor': 2} if num_workers > 0 else {}),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
