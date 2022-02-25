from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
import json
import pickle
import pandas as pd
import joblib
import sklearn
import xgboost as xgb
import gc
import pre_process_num
import pre_process_date
import os
import csv


def pre_process_num(df):#Pass the dataframe
	import pandas as pd
	chunk_num=1
	chunk_processed=None
	df_num_test_final=None
	#Pick most important numerical features:--
	num_fea=['Id', 'L3_S38_F3960', 'L3_S38_F3956', 'L3_S38_F3952', 'L3_S32_F3850', 'L3_S30_F3554', 'L3_S29_F3442', 'L3_S30_F3754', 'L3_S30_F3774', 'L3_S35_F3896', 'L3_S30_F3749', 'L3_S30_F3574', 'L1_S24_F1723', 'L3_S29_F3330', 'L3_S30_F3819', 'L3_S30_F3519', 'L3_S33_F3865', 'L1_S24_F1846', 'L3_S30_F3794', 'L3_S29_F3479', 'L3_S30_F3809', 'L1_S24_F1695', 'L3_S29_F3333', 'L3_S30_F3804', 'L3_S29_F3449', 'L3_S33_F3859', 'L3_S29_F3336', 'L3_S29_F3430', 'L3_S29_F3424', 'L3_S30_F3769', 'L3_S30_F3759', 'L3_S29_F3382', 'L3_S29_F3436', 'L3_S29_F3370', 'L3_S29_F3373', 'L3_S29_F3354', 'L3_S29_F3427', 'L3_S30_F3829', 'L3_S29_F3315', 'L0_S1_F28', 'L3_S29_F3395', 'L3_S30_F3744', 'L3_S30_F3499', 'L3_S30_F3534', 'L3_S29_F3324', 'L3_S36_F3924', 'L3_S29_F3401', 'L3_S29_F3327', 'L3_S30_F3569', 'L3_S33_F3863', 'L3_S30_F3784', 'L3_S29_F3345', 'L3_S30_F3609', 'L3_S29_F3376', 'L3_S29_F3367', 'L3_S36_F3920', 'L3_S33_F3855', 'L3_S30_F3799', 'L3_S29_F3388', 'L0_S0_F20', 'L3_S30_F3539', 'L3_S29_F3318', 'L0_S0_F0', 'L3_S30_F3494', 'L3_S30_F3604', 'L3_S30_F3764', 'L3_S33_F3857', 'L3_S30_F3709', 'L3_S29_F3379', 'L3_S30_F3559', 'L3_S30_F3514', 'L3_S30_F3639', 'L3_S30_F3669', 'L3_S29_F3452', 'L3_S30_F3579', 'L0_S0_F2', 'L3_S30_F3629', 'L3_S30_F3649', 'L3_S29_F3342', 'L3_S29_F3455', 'L0_S9_F160', 'L3_S29_F3458', 'L3_S30_F3544', 'L0_S0_F18', 'L3_S30_F3589', 'L3_S29_F3439', 'L3_S29_F3351', 'L3_S29_F3348', 'L3_S30_F3689', 'L0_S11_F290', 'L3_S29_F3404']
	
	#Creating a blank place holder for Dataframe of desired size
	df_num_test_final=pd.DataFrame(columns=num_fea)
	
	# Setting initial value of the counter to zero
	rowcount  = 0
	#iterating through the whole file
	for row in open(df):
		rowcount+= 1
	print("Row count",rowcount)
	
	# for chunk in pd.read_csv(df,chunksize=rowcount//10):
	for chunk in pd.read_csv(df,chunksize=rowcount//10):
		try:
			chunk=chunk[num_fea]
		except:
			print("Missing Features in input file!!")
			break
		print("\n\n")
		print("Processing Test set, Chunk",chunk_num)
		chunk_processed=chunk.fillna(0)
		df_num_test_final=pd.concat([df_num_test_final,chunk_processed])
		chunk_num+=1

	# chunk=pd.read_csv(df)
	# df_num_test_final=chunk[num_fea]
	return df_num_test_final