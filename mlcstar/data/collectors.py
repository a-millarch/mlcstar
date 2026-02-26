import os
import pandas as pd
import pyarrow.parquet as pq
import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Datastore, Dataset, Environment

from mlcstar.utils import is_file_present, are_files_present
from mlcstar.utils import get_cfg, get_base_df
from mlcstar.utils import logger


def download_to_local(FILES: list, LOCAL_DIR="data/dl"):
    """Download parquet files from Azure datastore to local directory."""
    WS = Workspace.from_config()
    datastore_sp = Datastore.get(WS, "sp_data")  # TODO: update datastore name if different

    total_failed = 0
    os.makedirs(LOCAL_DIR, exist_ok=True)
    f_status = dict()
    for fn in FILES:
        if is_file_present(f"data/dl/CPMI_{fn}.parquet"):
            pass
        else:
            try:
                print("> ", fn)
                ds = Dataset.File.from_files((datastore_sp, "CPMI_" + fn + ".parquet"))
                print(">> Downloading...")
                ds.download(LOCAL_DIR)
                print(">> Done!")
                f_status[fn] = True
            except Exception:
                f_status[fn] = False
                total_failed += 1
                print(">> Failed!!!")
                print(f"Could not load {fn}!", False)


def collect_subsets(cfg, base=None):
    """Collect all raw data subsets from Azure to local CSV files."""
    # First, load small files using population filter function
    for filename in cfg['default_load_filenames']:
        if is_file_present(f"data/raw/{filename}.csv"):
            logger.info(f'{filename} found in raw')
        else:
            population_filter_parquet(filename, base=base)

    # Resolve parquet directory: external or local
    use_external = cfg.get('use_external_dl', False)
    if use_external:
        dl_dir = cfg['external_dl_path']
        logger.info(f'Using external dl directory: {dl_dir}')
    else:
        dl_dir = 'data/dl'

    # For larger files, download parquet first (skip if using external)
    if are_files_present(dl_dir,
                         ['CPMI_' + i for i in cfg['large_load_filenames']],
                         extension='.parquet'):
        logger.info('parquet files found, continue')
    else:
        if use_external:
            raise FileNotFoundError(
                f"use_external_dl is true but parquet files not found in {dl_dir}"
            )
        logger.info('missing local parquet files, downloading')
        download_to_local(cfg['large_load_filenames'])

    # Chunk filter to only population
    for filename in cfg['large_load_filenames']:
        if is_file_present(f"data/raw/{filename}.csv"):
            logger.info(f'{filename} found in raw')
        else:
            logger.info(f'Processing {filename}')
            chunk_filter_parquet(filename, base=base, dl_dir=dl_dir)


def chunk_filter_parquet(filename, base=None, chunk_size=4000000, dl_dir='data/dl'):
    """Filter a large parquet file to only population patients, chunk by chunk."""
    if base is None:
        base = get_base_df()
        logger.info("Loaded base df")

    poplist = base['CPR_hash'].unique()
    file_path = os.path.join(dl_dir, f'CPMI_{filename}.parquet')
    output_path = f'data/raw/{filename}.csv'

    parquet_file = pq.ParquetFile(file_path)

    chunk_n = 0
    num_chunks = (parquet_file.metadata.num_rows / chunk_size)
    logger.info(f'>Initiating {num_chunks} chunks')
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk_n = chunk_n + 1
        print(f">>{chunk_n} of {num_chunks}chunks", end='\r')
        chunk_df = batch.to_pandas()
        chunk_df = chunk_df[chunk_df.CPR_hash.isin(poplist)]
        chunk_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    logger.info(f'Finished, saved file at: {output_path}')


def population_filter_parquet(filename, base=None, blobstore_uri=None):
    from mlcstar.utils import get_cfg
    cfg= get_cfg()
    """Download a parquet file and filter to only population patients."""
    if base is None:
        base = get_base_df()
    if blobstore_uri is None:
        blobstore_uri = cfg.get("raw_file_path", "")  # TODO: set blobstore_uri in config

    logger.info(f'Collecting and filtering {filename}')
    path = f'{blobstore_uri}CPMI_{filename}.parquet'
    ds = Dataset.Tabular.from_parquet_files(path=path)
    df = ds.to_pandas_dataframe()
    df = df[df.CPR_hash.isin(base.CPR_hash)]
    logger.info(f'loaded {len(df)} rows. Saving file.')
    df.to_csv(f"data/raw/{filename}.csv")

def get_sp_data(file, n_take:int|None=None, filter_function=None):
    """
    Fetches parquet data from SP-data Azure Blob Storage, with optional row sampling and filtering.

    Args:
        file (str): Filename identifier (will be formatted as CPMI_{file}.parquet)
        n_take (int, optional): Number of rows to retrieve. Defaults to full dataset.
        filter_function (callable, optional): Function to filter retrieved DataFrame

    Returns:
        pandas.DataFrame: Processed data from specified file

    Example:
    # Take 1000 rows from PatientInfo where Sex is Male
        >>> df = get_sp_data('PatientInfo', n_take=1000)
        >>> df_filtered = get_sp_data('sales', filter_function=lambda x: x[x['Køn'] == 'Mand'])
    """
   
    path = f'https://forskerpln0ybkrdls01.blob.core.windows.net/sp-data/CPMI_{file}.parquet'
    ds = Dataset.Tabular.from_parquet_files(path=path)
    if n_take:
            df = ds.take(n_take).to_pandas_dataframe()
    else:
        df =ds.to_pandas_dataframe()
        
    if filter_function is None:
        pass
    else:
        df = filter_function(df)
    return df