"functions that calculate metrics to check data integrity"

import pandas as pd 

def calculate_duplication(df):
	duplicated_rows = df[df.duplicated(keep='first')]
	duplicated_rows_count = duplicated_rows.count()
	count = duplicated_rows_count[0]
	rows = duplicated_rows
	return count, rows

def calculate_missing_value_in_columns(df):
	output = {}
	for item in df.columns:
		output[item] = {
			'count': len(df[item]),
			'missing': df[item].isna().sum(), 
			'pc_missing': round((df[item].isna().sum() / len(df[item])) , 4)
			}
 
	return output
	
