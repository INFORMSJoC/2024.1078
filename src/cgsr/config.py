import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class ExperimentConfig:
    model: str = "RLGD"
    dataset: str = "Douban-Book"
    seed: int = 2022
    reproducibility: bool = True

    data_path: str = "data"
    result_path: str = "results"
    gpu_id: str = field(default_factory=lambda: os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    use_gpu: bool = True
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    USER_ID_FIELD: str = "user_id"
    ITEM_ID_FIELD: str = "item_id"
    NEG_PREFIX: str = "neg_"
    NET_SOURCE_ID_FIELD: str = "source_id"
    NET_TARGET_ID_FIELD: str = "target_id"

    embedding_size: int = 32
    n_layers: int = 2
    epochs: int = 200
    train_batch_size: int = 2048
    eval_batch_size: int = 4096
    eval_step: int = 1
    stopping_step: int = 10
    learner: str = "adam"
    learning_rate: float = 0.0007
    weight_decay: float = 0.0
    loss_decimal_place: int = 4
    shuffle: bool = True
    enable_amp: bool = False
    enable_scaler: bool = False
    show_progress: bool = True

    metrics: tuple = ("recall", "mrr", "ndcg", "hit", "precision")
    topk: tuple = (1, 3, 5, 10, 20)
    valid_metric: str = "recall@10"
    metric_decimal_place: int = 4
    split_ratios: tuple = (0.7, 0.1, 0.2)

    train_recommender: bool = True
    train_generator: bool = True
    cf_pos_flag: bool = True
    cf_neg_flag: bool = False
    cf_loss_function: str = "BPR"
    ib_beta: float = 1.0
    n_part_layers: int = 1
    gamma: float = 0.95
    glr: float = 0.04
    ssl_reg: float = 1e-5
    reg_weight: float = 1e-5
    max_neighbor_size: int = 16
    replace_step: int = 9
    replace_num: int = 800
    cf_pos_pos_weight: float = 0.5
    cf_pos_neg_weight: float = 0.05
    generator: str = "GCN"
    cf_generate_type: str = "max"

    train_neg_sample_args: dict = field(
        default_factory=lambda: {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
    )

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Run the CGSR Douban-Book experiment.")
        parser.add_argument("--gpu_id", default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--eval_step", type=int, default=None)
        parser.add_argument("--show_progress", type=str, default=None)
        parser.add_argument("--data_path", default=None)
        parser.add_argument("--result_path", default=None)
        args = parser.parse_args()

        config = cls()
        for key in ("gpu_id", "epochs", "eval_step", "data_path", "result_path"):
            value = getattr(args, key)
            if value is not None:
                setattr(config, key, value)
        if args.show_progress is not None:
            config.show_progress = args.show_progress.lower() in {"1", "true", "yes", "y"}
        config.init_device()
        return config

    def init_device(self):
        self.gpu_id = str(self.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        if self.use_gpu and torch.cuda.is_available() and str(self.gpu_id) != "":
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

    @property
    def dataset_dir(self):
        return Path(self.data_path) / self.dataset

    @property
    def valid_metric_bigger(self):
        return True

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def as_dict(self):
        data = asdict(self)
        data["device"] = str(self.device)
        return data
