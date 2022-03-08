from wsgiref import simple_server
from flask import Flask, request, render_template,make_response
from flask import Response,send_file
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


print(xgb.__version__)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)




@app.route("/", methods=['GET'])
# @app.route('/<path:path>')
@cross_origin()
def home():
    # file_path = request.form.get("file_path")
    file=request.files.get("file")
    # file=request.files['csvfile']
    # # print('Home path',file_path)
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():

    try:
        if request.json is not None:
            file=request.files.get("file")
            # print('path',file_path)

        elif request.form is not None:

            # num_fea=['Id', 'L3_S38_F3960', 'L3_S38_F3956', 'L3_S38_F3952', 'L3_S32_F3850', 'L3_S30_F3554', 'L3_S29_F3442', 'L3_S30_F3754', 'L3_S30_F3774', 'L3_S35_F3896', 'L3_S30_F3749', 'L3_S30_F3574', 'L1_S24_F1723', 'L3_S29_F3330', 'L3_S30_F3819', 'L3_S30_F3519', 'L3_S33_F3865', 'L1_S24_F1846', 'L3_S30_F3794', 'L3_S29_F3479', 'L3_S30_F3809', 'L1_S24_F1695', 'L3_S29_F3333', 'L3_S30_F3804', 'L3_S29_F3449', 'L3_S33_F3859', 'L3_S29_F3336', 'L3_S29_F3430', 'L3_S29_F3424', 'L3_S30_F3769', 'L3_S30_F3759', 'L3_S29_F3382', 'L3_S29_F3436', 'L3_S29_F3370', 'L3_S29_F3373', 'L3_S29_F3354', 'L3_S29_F3427', 'L3_S30_F3829', 'L3_S29_F3315', 'L0_S1_F28', 'L3_S29_F3395', 'L3_S30_F3744', 'L3_S30_F3499', 'L3_S30_F3534', 'L3_S29_F3324', 'L3_S36_F3924', 'L3_S29_F3401', 'L3_S29_F3327', 'L3_S30_F3569', 'L3_S33_F3863', 'L3_S30_F3784', 'L3_S29_F3345', 'L3_S30_F3609', 'L3_S29_F3376', 'L3_S29_F3367', 'L3_S36_F3920', 'L3_S33_F3855', 'L3_S30_F3799', 'L3_S29_F3388', 'L0_S0_F20', 'L3_S30_F3539', 'L3_S29_F3318', 'L0_S0_F0', 'L3_S30_F3494', 'L3_S30_F3604', 'L3_S30_F3764', 'L3_S33_F3857', 'L3_S30_F3709', 'L3_S29_F3379', 'L3_S30_F3559', 'L3_S30_F3514', 'L3_S30_F3639', 'L3_S30_F3669', 'L3_S29_F3452', 'L3_S30_F3579', 'L0_S0_F2', 'L3_S30_F3629', 'L3_S30_F3649', 'L3_S29_F3342', 'L3_S29_F3455', 'L0_S9_F160', 'L3_S29_F3458', 'L3_S30_F3544', 'L0_S0_F18', 'L3_S30_F3589', 'L3_S29_F3439', 'L3_S29_F3351', 'L3_S29_F3348', 'L3_S30_F3689', 'L0_S11_F290', 'L3_S29_F3404']

            #Request an input file:-
            print("Getting CSV file....")
            file=request.files.get("file")


            print("Reading csvfile for num features...")
            print(type(file))
            try:
                df=pd.read_csv(file)
                df=df.astype(float)
                df=df.astype({"Id":int})

            except:
                # return "Input is not readable format!"
                raise Exception("Input is not readable format!")


            #Raise error if data point count exceeds limit set (this is limited by deployment platform chosen, which requires faster API response)
            if len(df)>20:
               raise Exception("Ensure number of datapoints is 20 or less")


            #Preprocessing numerical:--
            df_num=pre_process_num.pre_process_num(df)
            print(type(df_num))




            print("Reading csv file for date features...")
            print(type(file))
            # df=pd.read_csv(file)
            df_date=pre_process_date.pre_process_date(df,file)
            print(type(df_date))


            #Merge_datasets:
            print("Merging processed dataset...")
            final_test=pd.merge(df_date,df_num,on="Id",how="inner")



            #Select only those features which were used during model training:--
            train_features=['L0_S0_F0', 'L0_S0_F2', 'L0_S0_F18', 'L0_S0_F20', 'L0_S1_F28', 'L0_S9_F160', 'L0_S11_F290', 'L1_S24_F1695', 'L1_S24_F1723', 'L1_S24_F1846', 'L3_S29_F3315', 'L3_S29_F3318', 'L3_S29_F3324', 'L3_S29_F3327', 'L3_S29_F3330', 'L3_S29_F3333', 'L3_S29_F3336', 'L3_S29_F3342', 'L3_S29_F3345', 'L3_S29_F3348', 'L3_S29_F3351', 'L3_S29_F3354', 'L3_S29_F3367', 'L3_S29_F3370', 'L3_S29_F3373', 'L3_S29_F3376', 'L3_S29_F3379', 'L3_S29_F3382', 'L3_S29_F3388', 'L3_S29_F3395', 'L3_S29_F3401', 'L3_S29_F3404', 'L3_S29_F3424', 'L3_S29_F3427', 'L3_S29_F3430', 'L3_S29_F3436', 'L3_S29_F3439', 'L3_S29_F3442', 'L3_S29_F3449', 'L3_S29_F3452', 'L3_S29_F3455', 'L3_S29_F3458', 'L3_S29_F3479', 'L3_S30_F3494', 'L3_S30_F3499', 'L3_S30_F3514', 'L3_S30_F3519', 'L3_S30_F3534', 'L3_S30_F3539', 'L3_S30_F3544', 'L3_S30_F3554', 'L3_S30_F3559', 'L3_S30_F3569', 'L3_S30_F3574', 'L3_S30_F3579', 'L3_S30_F3589', 'L3_S30_F3604', 'L3_S30_F3609', 'L3_S30_F3629', 'L3_S30_F3639', 'L3_S30_F3649', 'L3_S30_F3669', 'L3_S30_F3689', 'L3_S30_F3709', 'L3_S30_F3744', 'L3_S30_F3749', 'L3_S30_F3754', 'L3_S30_F3759', 'L3_S30_F3764', 'L3_S30_F3769', 'L3_S30_F3774', 'L3_S30_F3784', 'L3_S30_F3794', 'L3_S30_F3799', 'L3_S30_F3804', 'L3_S30_F3809', 'L3_S30_F3819', 'L3_S30_F3829', 'L3_S32_F3850', 'L3_S33_F3855', 'L3_S33_F3857', 'L3_S33_F3859', 'L3_S33_F3863', 'L3_S33_F3865', 'L3_S35_F3896', 'L3_S36_F3920', 'L3_S36_F3924', 'L3_S38_F3952', 'L3_S38_F3956', 'L3_S38_F3960', 'L0_S0', 'L0_S1', 'L0_S2', 'L0_S3', 'L0_S4', 'L0_S5', 'L0_S6', 'L0_S7', 'L0_S8', 'L0_S9', 'L0_S10', 'L0_S11', 'L0_S12', 'L0_S13', 'L0_S14', 'L0_S15', 'L0_S16', 'L0_S17', 'L0_S18', 'L0_S19', 'L0_S20', 'L0_S21', 'L0_S22', 'L0_S23', 'L1_S24', 'L1_S25', 'L2_S26', 'L2_S27', 'L2_S28', 'L3_S29', 'L3_S30', 'L3_S31', 'L3_S32', 'L3_S33', 'L3_S34', 'L3_S35', 'L3_S36', 'L3_S37', 'L3_S38', 'L3_S39', 'L3_S40', 'L3_S41', 'L3_S42', 'L3_S43', 'L3_S44', 'L3_S45', 'L3_S46', 'L3_S47', 'L3_S48', 'L3_S49', 'L3_S50', 'L3_S51', 'Total_time_taken', 'station_32', 'mindate', 'maxdate', 'min_time_station', 'max_time_station', 'min_Id_rev', 'min_Id', 'start_station', 'end_station']

            x_test=final_test.drop(["Id"],axis=1)
            x_test=x_test.fillna(0)
            x_test=x_test[list(train_features)]

            #Predict on merged dataset:--
            print("Loading the model...")
            filename = 'xg_boost_date_num1.pickle'

            loaded_model = pickle.load(open(filename, 'rb'))
            print("Predicting....")
            df_test_pred = loaded_model.predict(x_test)    

            #Predicted dataframe:--
            print("List of predictions....")
            result = list(df_test_pred)

            print("Dataframe of predictions....")
            result = pd.DataFrame(list(zip(final_test.Id, result)), columns=['Part ID', 'Prediction'])
            result['Prediction'] = result['Prediction'].map({1:'Defect likely', 0:'Defect Unlikely'})

            global result_df
            result_df=result
            print_table=result.head(5)

            print("Sending .csv file....")
            resp = make_response(result_df.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=result.csv"
            resp.headers["Content-Type"] = "text/csv"
            
            # render_template('index.html',tables=[print_table.to_html(classes='data')], titles=print_table.columns.values)


            return resp


        else:
        	print("No match found!!")

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


# # @app.route("/return_file", methods=['POST'])
# # @cross_origin()
# def return_file(result_df):

#     try:
#         print("Sending .csv file....")
#         resp = make_response(result_df.to_csv())
#         resp.headers["Content-Disposition"] = "attachment; filename=result.csv"
#         resp.headers["Content-Type"] = "text/csv"
#         return resp

#     except ValueError:
#         return Response("Error Occurred! %s" % ValueError)
#     except KeyError:
#         return Response("Error Occurred! %s" % KeyError)
#     except Exception as e:
#         return Response("Error Occurred! %s" % e)


port = int(os.getenv("PORT", 8000))
if __name__ == "__main__":
    result_df=None
    app.run(debug=True)

