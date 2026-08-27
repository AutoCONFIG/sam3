# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from typing import Callable, Iterable, Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset


class TorchDataset:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        enable_distributed_sampler=True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        # persistent_workers/prefetch_factor 只在多进程加载下合法
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self._loader = None
        assert not isinstance(self.dataset, IterableDataset), "Not supported yet"
        if enable_distributed_sampler:
            self.sampler = DistributedSampler(self.dataset, shuffle=self.shuffle)
        else:
            # pyrefly: ignore [bad-assignment]
            self.sampler = None

    def get_loader(self, epoch) -> Iterable:
        if self.sampler:
            self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            # pyrefly: ignore [missing-attribute]
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # persistent 模式下复用同一个 DataLoader, worker 跨 epoch 常驻,
        # 省掉每 epoch 重建 worker 进程 + 重新 fork 数据集的开销。
        # 注意: worker 持有 fork 时的数据集副本, 之后主进程对 dataset 的修改
        # (如 curr_epoch) 不会同步给 worker — 仅适用于不依赖 epoch 的变换
        # (模板用的 RandomResizeAPI 即如此; ScheduledRandomResizeAPI 不适用)。
        if self.persistent_workers and self._loader is not None:
            return self._loader

        kwargs = {}
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=self.sampler,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
            **kwargs,
        )
        if self.persistent_workers:
            self._loader = loader
        return loader
