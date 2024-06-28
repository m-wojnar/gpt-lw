import random

from cfg_dataset.cfg import CFG

DELIM_TOKEN_CFG = ","


if __name__ == "__main__":
    cfg_name = "cfg3b_half"
    n_train_samples = 100000
    n_val_samples = 1000
    dataset_name = f"{cfg_name}_{n_train_samples}"
    log_stats = True

    random.seed(42)
    cfg = CFG(rules_file=f"configs/cfg/{cfg_name}.cfg")

    print(f"Generating dataset for CFG {cfg_name} with {n_train_samples} samples...")
    samples = cfg.sample_rand(n=n_train_samples)

    unique_samples = list(set(samples))
    print(f"Number of unique samples: {len(unique_samples)}")

    val_samples = unique_samples[:n_val_samples]
    train_samples = [s for s in samples if s not in val_samples]
    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of val samples: {len(val_samples)}")

    train_dataset_str = DELIM_TOKEN + DELIM_TOKEN.join(train_samples) + DELIM_TOKEN
    val_dataset_str = DELIM_TOKEN + DELIM_TOKEN.join(val_samples) + DELIM_TOKEN

    if log_stats:
        avg_len = sum([len(s) for s in train_samples]) / len(train_samples)
        n_tokens = len(train_dataset_str)
        vocab_size = len(set(train_dataset_str))
        dataset_size_gb = n_tokens * 1e-9 # 1 char = 1 byte

        print(f"Train dataset stats for CFG {cfg_name} with {len(train_samples)} samples:")
        print(f"Average length: {avg_len}")
        print(f"Number of tokens: {n_tokens}")
        print(f"Vocabulary size: {vocab_size}")

    splits = [("train", train_dataset_str), ("val", val_dataset_str)]
    for split_name, dataset_str in splits:
        dataset_file = f"cfg_dataset/{dataset_name}_{split_name}.txt"
        print(f"Writing {split_name} dataset to file {dataset_file}")
        with open(dataset_file, "w") as f:
            f.write(dataset_str)
