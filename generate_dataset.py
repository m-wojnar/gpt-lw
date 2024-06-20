from tqdm import tqdm
from cfg_dataset.cfg import CFG


if __name__ == "__main__":
  cfg_name = "simple4"
  n_samples = 100000
  delim_token = ","
  dataset_file = f"cfg_dataset/{cfg_name}_{n_samples}.txt"
  log_stats = True

  cfg = CFG(rules_file=f"cfg_dataset/configs/{cfg_name}.cfg")

  print(f"Generating dataset for CFG {cfg_name} with {n_samples} samples...")
  samples = [cfg.sample() for _ in tqdm(range(n_samples))]
  dataset_str = delim_token.join(samples)

  if log_stats:
    avg_len = sum([len(s) for s in samples]) / n_samples
    n_tokens = len(dataset_str)
    vocab_size = len(set(dataset_str))
    dataset_size_gb = n_tokens * 1e-9 # 1 char = 1 byte

    print(f"Dataset stats for CFG {cfg_name} with {n_samples} samples:")
    print(f"Average length: {avg_len}")
    print(f"Number of tokens: {n_tokens}")
    print(f"Vocabulary size: {vocab_size}")

  print(f"Writing dataset to file {dataset_file}")
  with open(dataset_file, "w") as f:
    f.write(dataset_str)