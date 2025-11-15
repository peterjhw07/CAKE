"""CAKE Miscellaneous Functions"""

# Imports
import pandas as pd


# Read imported data
def import_data(filename, sheet_name, t_col, col, temp_col=None):
    if '.xlsx' in filename:
        df = pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl', dtype=str)
        headers = list(pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl').columns)
    elif '.csv' in filename or '.txt' in filename:
        df = pd.read_csv(filename)
        headers = list(pd.read_csv(filename).columns)
    else:
        try:
            df = pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl', dtype=str)
            headers = list(pd.read_excel(filename, sheet_name=sheet_name, engine='openpyxl').columns)
        except Exception as e:
            raise e
    if isinstance(col, int): col = [col]
    conv_col = [i for i in [t_col, *col, temp_col] if i is not None]
    try:
        for i in conv_col:
            df[headers[i]] = pd.to_numeric(df[headers[i]], downcast='float')
        return df
    except ValueError:
        raise ValueError('Excel file must contain data rows (i.e. col specified)'
                         ' of numerical input with at most 1 header row.')
