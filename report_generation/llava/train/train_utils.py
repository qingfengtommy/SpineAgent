import os, json, logging, functools
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scan")

# ----------------------------------------------------------------------
# 1. load JSON (same as yours, but wrapped in a tiny helper)
# ----------------------------------------------------------------------
def load_json(path: str, checkpoint_key: str | None = None):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if checkpoint_key:
            data = data[checkpoint_key]
        log.info("Loaded JSON with %d patients.", len(data))
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Cannot open {path}: {e}") from e

# ----------------------------------------------------------------------
# 2. build *candidate* filename list (cheap string work, single-thread)
# ----------------------------------------------------------------------
def build_candidates(data, prefix):
    """
    Yields (full_path, patient_id, modality) for every candidate file.
    """
    modalities = {
        "t1":  ["t1", "t1_left"],
        "t2":  ["t2", "t2_left"],
    }
    for pid, mod_dict in data.items():
        for modality, sub_modalities in mod_dict.items():
            if modality == "unk":        # skip unknown
                continue
            for sub_mod, sub_paths in sub_modalities.items():
                for rel in sub_paths:
                    core = f"{pid}_{sub_mod}_{rel}".removesuffix(".png")
                    for suffix in modalities[modality]:
                        yield os.path.join(prefix, suffix, f"{core}.png"), pid, modality

# ----------------------------------------------------------------------
# 3. cheap I/O check – runs in *multiple* processes
# ----------------------------------------------------------------------
def file_exists(args):
    path, pid, modality = args
    return (pid, modality, path) if os.path.exists(path) else None

# ----------------------------------------------------------------------
# 4. main
# ----------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

def main():
    INPUT_JSON  = "/home/gongbsun/dinov2_spinal/out.json"
    OUTPUT_JSON = "/home/gongbsun/dinov2_spinal/full_paths.json"
    PREFIX      = "/home/ranzh8/mri_spinal"
    N_WORKERS   = min(32, (os.cpu_count() or 8))     # threads, not processes
    CHUNK_SIZE  = 50_000                             # ↔ memory / responsiveness

    data = load_json(INPUT_JSON)
    total = sum(
        len(v[sub_m])
        for v in data.values()           # just to get a progress total
        for sub_m in v.values() if sub_m != "unk"
    )
    log.info("Need to check %,d paths.", total)

    full_paths = defaultdict(lambda: defaultdict(list))
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool, \
         tqdm(total=total, unit="file") as bar:

        cand_iter = build_candidates(data, PREFIX)
        while True:
            batch = list(islice(cand_iter, CHUNK_SIZE))
            if not batch:
                break

            for (pid, modality, path) in pool.map(file_exists, batch):
                if path is None:          # file missing
                    continue
                full_paths[pid][modality].append(path)
                bar.update(1)

    log.info("Valid files: %,d",
             sum(len(v[m]) for v in full_paths.values() for m in v))

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(full_paths, f, indent=4)
    log.info("Wrote %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()