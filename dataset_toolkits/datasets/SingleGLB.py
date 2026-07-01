import os
import argparse
import pandas as pd
from tqdm import tqdm


def add_args(parser: argparse.ArgumentParser):
    pass


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    records = []
    rows = metadata.to_dict('records')
    for metadatum in tqdm(rows, desc=desc):
        sha256 = str(metadatum['sha256'])
        local_path = str(metadatum['local_path'])
        file_path = local_path if os.path.isabs(local_path) else os.path.join(output_dir, local_path)
        try:
            record = func(file_path, sha256)
            if record is not None:
                records.append(record)
        except Exception as exc:
            print(f"Error processing object {sha256}: {exc}", flush=True)
    return pd.DataFrame.from_records(records)
