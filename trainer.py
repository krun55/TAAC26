"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import ModelEMA, sigmoid_focal_loss
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_bf16_train: bool = False,
        ema_decay: float = 0.0,
        ema_start_epoch: int = 3,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.use_bf16_train: bool = use_bf16_train
        self.ema_decay: float = ema_decay
        self.ema_start_epoch: int = max(1, ema_start_epoch)
        self.ema: Optional[ModelEMA] = None
        if ema_decay > 0:
            logging.info(
                f"EMA scheduled: decay={ema_decay}, "
                f"start_epoch={self.ema_start_epoch}")
        if device.startswith('cuda'):
            self._autocast_device_type: Optional[str] = 'cuda'
        elif device == 'cpu':
            self._autocast_device_type = 'cpu'
        else:
            self._autocast_device_type = None
        self._use_train_autocast = (
            self.use_bf16_train and self._autocast_device_type is not None
        )

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")
        logging.info(
            f"use_bf16_train={self.use_bf16_train}, "
            f"train_autocast_enabled={self._use_train_autocast}, "
            f"autocast_device={self._autocast_device_type}")

    def _maybe_enable_ema(self, epoch: int) -> None:
        if self.ema_decay <= 0 or self.ema is not None:
            return
        if epoch < self.ema_start_epoch:
            return
        self.ema = ModelEMA(self.model, decay=self.ema_decay)
        logging.info(
            f"EMA enabled at epoch {epoch}: decay={self.ema_decay}")

    def _build_epoch_dir_name(self, epoch: int, global_step: int) -> str:
        """Build a checkpoint sub-directory name such as
        ``epoch1.global_step2500.layer=2.head=4.hidden=64``.
        """
        parts = [f"epoch{epoch}", f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        return ".".join(parts)

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config is None:
            raise ValueError("train_config is required when saving checkpoints")

        import json
        cfg_to_dump = self.train_config
        if ns_groups_copied:
            # Override the stored path to a filename relative to ckpt_dir;
            # infer.py already falls back to `<ckpt_dir>/<basename>` when
            # the recorded path is not absolute, which keeps the ckpt
            # portable across hosts.
            cfg_to_dump = dict(self.train_config)
            cfg_to_dump['ns_groups_json'] = os.path.basename(
                self.ns_groups_path)
        with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
            json.dump(cfg_to_dump, f, indent=2)

    def _save_epoch_checkpoint(
        self,
        epoch: int,
        global_step: int,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under an epoch checkpoint dir.

        Args:
            epoch: current epoch used to name the directory.
            global_step: current global step used to name the directory.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_epoch_dir_name(epoch, global_step)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        try:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            self._write_sidecar_files(ckpt_dir)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, saves each epoch checkpoint, and runs the
        periodic sparse re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            self._maybe_enable_ema(epoch)
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._save_epoch_checkpoint(epoch, total_step)

            # Between epochs, reinitialize high-cardinality sparse params
            # (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if (
                epoch < self.num_epochs
                and epoch >= self.reinit_sparse_after_epoch
                and self.sparse_optimizer is not None
            ):
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                if self.ema is not None:
                    self.ema.resync(self.model, reinit_ptrs)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_hours: Dict[str, torch.Tensor] = {}
        seq_dayofweeks: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_hours[domain] = device_batch.get(
                f'{domain}_hour',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_dayofweeks[domain] = device_batch.get(
                f'{domain}_dayofweek',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
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

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with torch.autocast(
            device_type=self._autocast_device_type or 'cpu',
            dtype=torch.bfloat16,
            enabled=self._use_train_autocast,
        ):
            logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)
        logits = logits.float()

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        When EMA is enabled, evaluation temporarily uses smoothed shadow
        weights and restores the raw training weights before returning.
        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        if self.ema is not None:
            self.ema.apply_shadow(self.model)
        try:
            self.model.eval()
            if not epoch:
                epoch = -1

            pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

            all_logits_list = []
            all_labels_list = []

            with torch.no_grad():
                for step, batch in pbar:
                    logits, labels = self._evaluate_step(batch)
                    all_logits_list.append(logits.detach().cpu())
                    all_labels_list.append(labels.detach().cpu())

            all_logits = torch.cat(all_logits_list, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0).long()

            # Binary AUC via sklearn.
            probs = torch.sigmoid(all_logits).numpy()
            labels_np = all_labels.numpy()

            # Filter NaN predictions (may appear if gradients explode).
            nan_mask = np.isnan(probs)
            if nan_mask.any():
                n_nan = int(nan_mask.sum())
                logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
                valid_mask = ~nan_mask
                probs = probs[valid_mask]
                labels_np = labels_np[valid_mask]

            if len(probs) == 0 or len(np.unique(labels_np)) < 2:
                auc = 0.0
            else:
                auc = float(roc_auc_score(labels_np, probs))

            # Binary logloss (same NaN filtering).
            valid_logits = all_logits[~torch.isnan(all_logits)]
            valid_labels = all_labels[~torch.isnan(all_logits)]
            if len(valid_logits) > 0:
                logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
            else:
                logloss = float('inf')

            return auc, logloss
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
