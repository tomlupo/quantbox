"""
Functions for file I/O operations including pickle serialization and Excel data
export
"""


import pickle
import pandas as pd
import os
import warnings
import pathlib

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    # Create directory if it doesn't exist
    path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# %% excel
def save_dict_to_excel(data,
                       output_dir="output",
                       index=True,
                       reset_index=False,
                       *args,
                       **kwargs):
    """Saves a dictionary of dictionaries to Excel files.

    Parameters:
        data (dict): Outer dictionary where each key is an Excel filename, and each value is another dictionary.
            Inner dictionary's keys are sheet names, and values are pandas DataFrames.
        output_dir (str): Directory to save the Excel files. Default is 'output'.
        index (bool): Whether to write row index to Excel. Default is True.
        reset_index (bool): Whether to reset the index before saving. Default is False.
        *args: Additional arguments to pass to pandas.DataFrame.to_excel.
        **kwargs: Additional keyword arguments to pass to pandas.DataFrame.to_excel.

    Returns:
        None: This function does not return anything.

    function version: 0.9
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # save each report to an excel file
    for excel_filename, sheets_dict in data.items():

        # create index sheet
        index_sheet = pd.DataFrame(index=pd.Index(sheets_dict.keys(), name="data_name"), data=sheets_dict.keys(),columns=["sheet_name"])

        # check sheet name lenghts
        sheet_name_lengths = [len(sheet_name) for sheet_name in sheets_dict.keys()]
        if max(sheet_name_lengths) > 31:
            # each sheet with a name longer than 31 characters will be renamed to name[:29]_i, etc.
            # create additional index sheet with the original sheet names mapping to new sheet names
            new_data = {}
            i = 1
            for k,v in sheets_dict.items():
                if len(k) > 31:
                    new_data[f"{k[:28]}_{i}"] = v
                    index_sheet.loc[k] = [f"{k[:28]}_{i}"]
                    i += 1
                else:
                    new_data[k] = v
            sheets_dict = new_data

        # add index sheet to data
        if not index and not reset_index:
            index_sheet = index_sheet.reset_index()
        sheets_dict["index_sheet"] = index_sheet

        # save each sheet to an excel file
        excel_path = os.path.join(output_dir, f"{excel_filename}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, df in sheets_dict.items():
                # Write each DataFrame to the specified sheet
                if reset_index:
                    df = df.reset_index()
                # check if columns are multiindex and index is not set to True
                if df.columns.nlevels > 1 and not index:
                    warnings.warn('Columns are multiindex but index is not set to True. Setting index to True.')
                    index = True
                df.to_excel(writer, sheet_name=sheet_name,index=index)

