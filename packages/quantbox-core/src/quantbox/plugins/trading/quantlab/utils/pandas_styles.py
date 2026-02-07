"""
Styling functions for pandas DataFrames to enhance visualization outputs
"""


import pandas as pd

def set_table_styles(dfs):
    index_names = {
        'selector': '.index_name',
        'props': 'background-color: #52527a; color: white;'
        }
    table = {
        'selector': 'table',
        'props': [
            ('padding', '6px'), 
            ]
        }
    
    headers = {
        'selector': 'th',
        'props': [
           ('background-color', '#9494b8'),
           ('color', 'black'),
           ('border-color', 'black'),
           ('border-style ', 'solid'),
           ('border-width','1px'),
           ('border-collapse', 'collapse'),
           ('padding', '5px'),
           ('-webkit-tetx-size-adjust', 'none')]
    }
    cells = {
        'selector': 'td',
        'props': [
           ('border-color', 'black'),
           ('border-style ', 'solid'),
           ('border-width','1px'),
           ('text-align', 'right'),
           ('padding', '5px'),
           ('-webkit-tetx-size-adjust', 'none')]
        }
    
    dfs = dfs.set_table_styles([headers,table,cells,index_names])
            
    return dfs

def set_format(df,
          pct_cols=[],
          thousands_cols=[],
          decimal_cols=[]):
    
    dfs = (
        df.style
        .format("{:,.0f}", thousands=' ', subset=thousands_cols)
        .format("{:.2f}", subset=decimal_cols)
        .format("{:.2%}", subset=pct_cols)   
        )
        
    dfs = dfs.pipe(set_table_styles)
    #dfs = dfs.apply(hide_nan)
    return dfs

def hide_nan(s):
    return ['color: white' if pd.isna(v[1]) else '' for v in s.iteritems()]