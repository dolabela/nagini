import pandas as pd 



def calculate_duplication(df):
	duplicated_rows = df[df.duplicated(keep='first')]
	duplicated_rows_count = duplicated_rows.count()
	count = duplicated_rows_count[0]
	rows = duplicated_rows
	return count, rows

def calculate_missin_value_in_columns(df):
	"""Return a Pandas dataframe describing the contents of a source dataframe including missing values."""

	variables = []
	count = []
	missing = []
	
	for item in df.columns:
		variables.append(item)
		count.append(len(df[item]))
		missing.append(df[item].isna().sum())

	output = pd.DataFrame({
		'variable': variables, 
		'count': count,
		'missing': missing, 
	})    

	return output
	

