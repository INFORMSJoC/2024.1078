import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

from cuda_env import configure_cuda_visible_devices

configure_cuda_visible_devices()

import numpy as np
import torch

from cgsr import CFPosGenerator, CGSR, CGSRTrainer, ExperimentConfig, prepare_data


def init_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def write_result(config, test_result, elapsed):
    result_dir = Path(config.result_path) / config.dataset
    result_dir.mkdir(parents=True, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    result_file = result_dir / f"{config.model}_{config.dataset}_{current_time}.txt"

    with result_file.open("w", encoding="utf-8") as f:
        f.write(f"model:{config.model}\n")
        f.write(f"dataset:{config.dataset}\n")
        f.write(f"learning_rate:{config.learning_rate}\n")
        f.write(f"epochs:{config.epochs}\n")
        f.write("K\trecall\tNDCG\tprecision\n")
        for k in (1, 3, 5, 10):
            f.write(
                f"{k}\t"
                f"{test_result[f'recall@{k}']}\t"
                f"{test_result[f'ndcg@{k}']}\t"
                f"{test_result[f'precision@{k}']}\n"
            )
        f.write(f"total_time:{elapsed:.2f}s\n")
    return result_file


def main():
    start_time = time.time()
    config = ExperimentConfig.from_args()
    init_seed(config.seed, config.reproducibility)

    dataset, train_data, valid_data, test_data = prepare_data(config)

    raw_neighbors, _ = dataset.user_neighbors(config.max_neighbor_size)

    init_seed(config.seed, config.reproducibility)
    model = CGSR(config, train_data.dataset).to(config.device)
    cf_pos_generator = CFPosGenerator(config, train_data).to(config.device)

    trainer = CGSRTrainer(config, train_data, model, raw_neighbors, cf_pos_generator)
    trainer.fit(
        train_data,
        valid_data,
        show_progress=config.show_progress,
    )
    test_result = trainer.evaluate(test_data)

    elapsed = time.time() - start_time
    result_file = write_result(config, test_result, elapsed)

    print("result_file", result_file)


if __name__ == "__main__":
    main()
