

from nagini.data_integrity import *

def test_if_is_duplicated_count():
	data = [{  "calories": 420, "duration": 50 },
		{  "calories": 380, "duration": 40 },
		{  "calories": 390, "duration": 45 },
		{  "calories": 390, "duration": 45 },
		{  "calories": 390, "duration": 45 } ]
	df = pd.DataFrame(data)
	calculated_value, _ = calculate_duplication(df)
	assert calculated_value == 2

def test_if_not_is_duplicated_count():
	data = [{  "calories": 420, "duration": 50 },
		{  "calories": 380, "duration": 40 },
		{  "calories": 390, "duration": 45 }  ]
	df = pd.DataFrame(data)
	calculated_value, _ = calculate_duplication(df)
	assert calculated_value == 0

def test_missing_values():
	actual_data = [{  "calories": 420, "duration": 50 },
	{  "calories": 380, "duration": 40 },
	{  "duration": 45 },
	{  "calories": 390 },
	{  "calories": 390 } ]
	actual_df = pd.DataFrame(actual_data)
	actual_missing_values = calculate_missing_value_in_columns(actual_df)
	expected_missing_values_dict = {'calories': {'count':5, 'missing': 1, 'pc_missing': 0.2}, 
								  'duration': {'count':5, 'missing': 2, 'pc_missing': 0.4} }

	assert actual_missing_values == expected_missing_values_dict