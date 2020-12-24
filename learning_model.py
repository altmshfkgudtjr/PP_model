import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import joblib
import pickle
from generator import get_eyetracking_data

'''
	Eyetracking Result => Model => Pass/Non-Pass
'''

'''
	<data>
	
	Center	Blink	Left	Right	
[	8		5		81		6		]
				...
[	4		3		90		3		]

'''

'''
	<label>
	1: pass
	0: non-pass
'''

def tranin_model(n):
	if n < 10:
		print('Too few data counts. Please enter a number greater than 10.')
		return

	data, label = get_eyetracking_data(n)

	model = LogisticRegression()
	kfold = KFold(n_splits=4, random_state=1, shuffle=True)

	for train_index, test_index in kfold.split(data):
		x_train, x_test = data[train_index], data[test_index]
		y_train, y_test = label[train_index], label[test_index]

		model.fit(x_train, y_train)

		y_pred = model.predict(x_test)
		print('정확도 : {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))

	saved_model = pickle.dumps(model)
	joblib.dump(model, 'model.pkl')
	print("Saved Done!")
	return


'''
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, 
																shuffle=True, 
																random_state=0) 
'''

def predict_model(center, blink, left, right):
	load_model = joblib.load('model.pkl')
	data = np.array([[center, blink, left, right]])
	result = load_model.predict(data)
	return bool(result)
'''
'''


'''
	Model Load
'''
# load_model = joblib.load('model.pkl')