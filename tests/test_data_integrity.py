

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