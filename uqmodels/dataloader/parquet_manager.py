# This module contains functions used for parquet analysis tools
from collections.abc import Iterator, Sequence
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import polars as pl
import math 
from tqdm import tqdm
from os import listdir
import warnings
from os.path import join,isfile

def update_parquet_metadata(
    input_file: str,
    output_file: str,
    metadata_updates: dict
):
    """
    Met à jour les metadata d'un fichier Parquet à partir d'un dict structuré :
    {
        "global": {...},
        "columns": {
            "col": {...},
            ...
        }
    }
    """

    pf = pq.ParquetFile(input_file)
    schema = pf.schema_arrow

    global_updates = metadata_updates.get("global", {})
    column_updates = metadata_updates.get("columns", {})

    # --- 1. Metadata globales ---
    file_metadata = dict(schema.metadata) if schema.metadata else {}
    for k, v in global_updates.items():
        file_metadata[k.encode()] = str(v).encode()

    # --- 2. Metadata colonnes ---
    new_fields = []
    schema_column_names = [f.name for f in schema]

    for field in schema:
        md = dict(field.metadata) if field.metadata else {}

        if field.name in column_updates:
            for k, v in column_updates[field.name].items():
                md[k.encode()] = str(v).encode()

        new_fields.append(field.with_metadata(md))

    # Warn colonnes inexistantes
    for col in column_updates:
        print("colonnne name", col)
        if col not in schema_column_names:
            warnings.warn(f"La colonne '{col}' n'existe pas dans le Parquet.")

    new_schema = pa.schema(new_fields, metadata=file_metadata)

    # --- 3. Réécriture row-group par row-group ---
    print( "Update each row group . .")
    with pq.ParquetWriter(output_file, new_schema) as writer:
        for i in tqdm(range(pf.num_row_groups)):
            writer.write_table(pf.read_row_group(i))

    print(f"✅ Parquet file updated : {output_file}")
    
def read_parquet_metadata(filename: str) -> dict:
    """
    Lit toutes les metadata d'un fichier Parquet et retourne un dict structuré :
    {
        "global": {...},
        "columns": {
            "col": {...}
        }
    }
    """
    pf = pq.ParquetFile(filename)
    schema = pf.schema_arrow

    result = {
        "global": {},
        "columns": {}
    }

    # --- Metadata globales ---
    if schema.metadata:
        for k, v in schema.metadata.items():
            result["global"][k.decode()] = v.decode()

    # --- Metadata colonnes ---
    for field in schema:
        col_md = {}
        if field.metadata:
            for k, v in field.metadata.items():
                col_md[k.decode()] = v.decode()
        result["columns"][field.name] = col_md

    return result

def get_column_list_from_concept(file_path,concepts_list):
     meta_dict=read_parquet_metadata(file_path)
     print(meta_dict)

     list_columns_with_concepts=[col for col in meta_dict["columns"].keys() if "concepts" in meta_dict["columns"][col].keys()]    
     return [x for x in list_columns_with_concepts if len(set(eval(meta_dict["columns"][x]["concepts"])).intersection(set(concepts_list)))>=1]
    
   
def build_filter_from_conditions(conditions: list) -> ds.Expression:
    """
    This function build a Pyarrow expression from a list of tuplet.
    [("col", ">", 30), ("col", "<", 65), ("country", "in", ["FR", "BE"])]
    Args:
        conditions (list) : List representing the conditions of the filter to create
        example :
            [("Timestamp", ">=", pd.Timestamp("2018-01-01")),
            ("Timestamp", "<", pd.Timestamp("2023-01-01")),
            ("FCU_MAT", "<=", 48),
            ("FCU_DAT","<=", 65)
            ]
    Return :
        An expression corresponding to the logical intersection of all conditions in tuplets
    """
    expr = None

    for column, op, value in conditions:
        field = ds.field(column)

        if op == "==":
            cond = field == value
        elif op == "!=":
            cond = field != value
        elif op == ">":
            cond = field > value
        elif op == ">=":
            cond = field >= value
        elif op == "<":
            cond = field < value
        elif op == "<=":
            cond = field <= value
        elif op == "in":
            cond = field.isin(value)
        else:
            raise ValueError(f"Opérateur non supporté : {op}")

        expr = cond if expr is None else expr & cond

    return expr

class optimizedDataLoader:
    """
    Stream filtered Parquet data without an initial counting scan.

    Parameters
    ----------
    dataset:
        PyArrow dataset.
    filter_expr:
        Predicate pushed down to the Parquet scanner.
    selected_columns:
        Columns to project.
    batch_size:
        Maximum number of rows returned per batch.
    """

    def __init__(
        self,
        dataset: ds.Dataset,
        filter_expr: ds.Expression | None = None,
        selected_columns: Sequence[str] | None = None,
        batch_size: int = 32_768,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")

        self.dataset = dataset
        self.filter_expr = filter_expr
        self.selected_columns = (
            list(selected_columns)
            if selected_columns is not None
            else None
        )
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[pl.DataFrame]:
        scanner = self.dataset.scanner(
            filter=self.filter_expr,
            columns=self.selected_columns,
            batch_size=self.batch_size,
            use_threads=True,
        )

        for record_batch in scanner.to_batches():
            if record_batch.num_rows == 0:
                continue

            yield pl.from_arrow(record_batch)
        
        
# Get dataset as polars table, pandas table, dataloader
def load_dataset(dataset_path, 
                filter_dict=None, 
                selected_columns=None, 
                selected_concepts=None, 
                batch_size=64,
                output_format="dataloader",
                postprocessing=None):
    """
        Load a dataset from backend storage, and return as output a polars , pandas dataframe or a dataloader. You can define a filter to
        avoid loading the full dataset in memory.

        Args:
            dataset_path (str): path to the dataset, (supported format are csv or parquet). you can pass a path to
            a single file, or a folder. In case of a folder, all file content are concatenated, but schema of every files in folder shall be the same.
            filter_dict (list): Filter to apply on rows, it is a list of intersected conditions on columns, For example:
                filter_dict = [
                    ("Timestamp", ">=", pd.Timestamp("2018-01-01")),
                    ("Timestamp", "<", pd.Timestamp("2023-01-01")),
                    ("FCU_MAT", "<=", 48),
                    ("FCU_DAT","<=", 65)
                    ]
                The output dataset will consider only rows with value on timestamp columns between the two given dates, and including conditions
                on column FCU_MAT, and FCU_DAT
            selected_columns (list): list of subset of columns to filter . By default if none, all columns are taken into account
            selected_concepts=["identifier","context"], # for now it suppose that concept field with always present in columns meta information if not-> crash of get_concept function
            batch_size (int): Size of the batch , when output_format is set as "dataloader"
            output_format (enum) : output format of the requested dataset --> ["polars", "pandas", "dataloader","pyarrow"]
    """
    
    dataset = ds.dataset(dataset_path)
    
    # Get list of column to filter from context list if it is not none
    
    if ".parquet"in dataset_path:
            parquet_path= dataset_path
    else:
        #TODO recuperer le path du premier parquet du dossier
        listfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
        parquet_path=listfiles[0]
    
    if selected_concepts:
       if selected_columns is None :
           selected_columns=[]
       selected_columns+= get_column_list_from_concept(parquet_path,selected_concepts)
    
    print("computed selected columns from contexts", selected_columns)
    # Conversion du dict en expression pyarrow
    
    if filter_dict:
        filter_dict = build_filter_from_conditions(filter_dict)

    if selected_columns==[]:
           selected_columns=None
    print("postprocesingv1",postprocessing)
    if output_format=="dataloader":
         return optimizedDataloader(
            pyarrow_dataset=dataset,
            filters=filter_dict,
            selected_columns=selected_columns,
            batch_size=batch_size,
            postprocessing=postprocessing
        )

    if output_format=="polars":
        if postprocessing:
            return postprocessing(pl.DataFrame(dataset.to_table(filter=filter_dict, columns=selected_columns)))
        else: 
            return pl.DataFrame(dataset.to_table(filter=filter_dict, columns=selected_columns))
    if output_format=="pandas":
        if postprocessing:
            return postprocessing(dataset.to_table(filter=filter_dict, columns=selected_columns).to_pandas())
        else:
            return dataset.to_table(filter=filter_dict, columns=selected_columns).to_pandas()
