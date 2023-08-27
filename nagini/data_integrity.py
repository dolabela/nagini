"functions that calculate metrics to check data integrity"

import pandas as pd 

def calculate_duplication(df):
	duplicated_rows = df[df.duplicated(keep='first')]
	duplicated_rows_count = duplicated_rows.count()
	count = duplicated_rows_count[0]
	rows = duplicated_rows
	return count, rows

def calculate_missing_value_in_columns(df):
	variables = []
	count = []
	missing = []
	pc_missing = []
	for item in df.columns:
		variables.append(item)
		count.append(len(df[item]))
		missing.append(df[item].isna().sum())
		pc_missing.append(round((df[item].isna().sum() / len(df[item])) , 4))
	output = pd.DataFrame({
		'variable': variables, 
		'count': count,
		'missing': missing, 
		'pc_missing': pc_missing
	})    
	return output
	

