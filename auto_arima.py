import argparse
import numpy as np
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

def auto_arima(in_csv_file_path):
	print('IN File==>' + in_csv_file_path)
	df = pd.read_csv(in_csv_file_path, header=0, infer_datetime_format=True, parse_dates=[0], index_col=[0])
	data_label = df.columns.values[0]
	print('data_label=' + data_label)

	#Perform test-train split at 80:20
	split_index = round(len(df)*0.8)
	split_date = df.index[split_index]
	df_train = df.loc[df.index <= split_date].copy()
	df_test = df.loc[df.index > split_date].copy()

	print('Training data set size='+str(len(df_train)))
	print('Test data set size='+str(len(df_test)))

	print('Starting auto_arima......')
	best_model = pm.auto_arima(df_train, start_p=0, start_q=0, max_p=3, max_q=3, m=1,
	    start_P=0, seasonal=False, d=1, D=0, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
	print(best_model.summary())

	best_model.fit(df_train)

	forecast = np.round(best_model.predict(n_periods=len(df_test)))
	print('Forecast points ==> ' + str(forecast))

	#compute the confidence intervals
	confidence = 0.9
	conf_lowers = []
	conf_uppers = []
	for i in range(len(forecast)):
		predicted = forecast[i]
		std_dev_sample = df_train.std()[0]
		t_value = t.ppf((1 + confidence) / 2, len(df_train) - 1)
		t_lower = std_dev_sample * t_value
		conf_lower = max(0, round(predicted - t_lower))
		conf_upper = round(predicted + t_lower)
		conf_lowers.append(conf_lower)
		conf_uppers.append(conf_upper)

	#now plot everything: the actuals, forecast and confidence intervals for the forecast

	prediction_label = 'Predicted'
	actual_label = 'Actual'
	conf_lower_label = 'Lower ' + str(confidence) + '% confidence limit'
	conf_upper_label = 'Upper ' + str(confidence) + '% confidence limit'

	df_test[prediction_label] = np.array(forecast)
	df_test[actual_label] = np.array(df_test[data_label])
	df_test[conf_lower_label] = np.array(conf_lowers)
	df_test[conf_upper_label] = np.array(conf_upper)
	df_all = pd.concat([df_train, df_test])

	df_all[[data_label, actual_label, prediction_label, conf_lower_label, conf_upper_label]].plot()
	plt.show()


# save this code to a file called auto_arima.py and run it as follows:
# python auto_arima.py --in_csv tb.csv
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--in_csv', required=True)
	args = parser.parse_args()

	auto_arima(args.in_csv)