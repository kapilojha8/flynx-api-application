# using flask_restful
from flask import Flask, jsonify, request

import mysql.connector
from sklearn.preprocessing import LabelEncoder
import pandas as pd  # Importing the pandas library for data manipulation and analysis
import numpy as np  # Importing the numpy library for numerical operations
import pickle  # Importing the pickle module for object serialization
import json

from flask import  render_template
# import jwt
import requests
import time
import datetime
import uuid
# import tableauserverclient as TSC
import config
import joblib



# creating the flask app
app = Flask(__name__)
app.static_folder = 'static'


# The purpose of this fucntion NumpyEncoder is to provide a custom JSON encoder (NumpyEncoder) that can handle NumPy arrays by 
# converting them to Python lists using the tolist() method before serializing them to JSON.
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def Connect_with_db_get_df():
    connection = mysql.connector.connect(
        host='localhost',
        user='ritik',
        password='MRpark@1234',
        database='MR_MT_DB'
    )

    # Create a cursor
    cursor = connection.cursor()

    # Execute a query
    query = "SELECT * FROM TireData"
    cursor.execute(query)

    # Fetch all rows
    rows = cursor.fetchall()

    # Get column names
    column_names = [i[0] for i in cursor.description]

    # Create a DataFrame
    df_raw = pd.DataFrame(rows, columns=column_names)

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Display the DataFrame
    return df_raw


@app.route('/n_months_predictions/', methods=['GET'])
def get_n_months_predictions():
    n_months_param = request.args.get('months')
    df_raw = Connect_with_db_get_df()
    raw_unique_all_Tire_ids = df_raw['Vehicle_Asset_Number'].unique()
    ThresholdTread = 1.6
    result_prediction = []
    for val in raw_unique_all_Tire_ids:
        # Extract the number of months from the payload 
        # Filter the 'df_raw' DataFrame for the current Tire ID
        # Calculate the average kilometers driven per month for the current Tire ID
        # Estimate the kilometers to be driven in the next 'N_months' based on the average
        # Predict the total kilometers driven after 'N_months'
        
        N_months = int(n_months_param)
        df = df_raw.loc[df_raw['Vehicle_Asset_Number'] == val].copy()
        avg_km_driven_p_month = (max(df['Km_driven'])+min(df['Km_driven']))/df.shape[0]
        km_for_next_n_month = N_months*avg_km_driven_p_month
        Predict_kms = max(df['Km_driven'])+km_for_next_n_month
        # Initialize a LabelEncoder instance
        # Apply label encoding to columns in the DataFrame
        le = LabelEncoder()
        df[['Country']] = df[['Country']].apply(lambda col1: le.fit_transform(col1))
        df[['Tire_Company']] = df[['Tire_Company']].apply(lambda col2: le.fit_transform(col2))
        df[['Site_Name']] = df[['Site_Name']].apply(lambda col2: le.fit_transform(col2))
        df[['Vehicle_Asset_Number']] = df[['Vehicle_Asset_Number']].apply(lambda col4: le.fit_transform(col4))
        df[['Tire Model']] = df[['Tire Model']].apply(lambda col5: le.fit_transform(col5))
        df[['Tire_Asset_Number']] = df[['Tire_Asset_Number']].apply(lambda col6: le.fit_transform(col6))
        df[['Road_Condition']] = df[['Road_Condition']].apply(lambda col9: le.fit_transform(col9))   
        
        df['Km_driven'] = pd.to_numeric(df['Km_driven'], errors='coerce')
        df['Tire_Tread'] = pd.to_numeric(df['Tire_Tread'], errors='coerce')
        df['Tire Model'] = pd.to_numeric(df['Tire Model'], errors='coerce')
        # Prepare the features (X) and target (y) for regression prediction
        X=df.drop(['Tire_Tread'],axis=1)
        y=df['Tire_Tread']
        # Take the first row of features and update the 'Km_driven' value with the predicted kilometers
        X = X.iloc[:1,:].copy()
        X = X.reset_index()
        X = X.drop("index",axis=1)
        X.at[0,'Km_driven']= float("% .2f"% Predict_kms)
        # Load the saved regression model
        with open('model_multi_linear.pkl', 'rb') as model:
            model = pickle.load(model)
            predicted_Tread = model.predict(X)
            need_to_replace = True if predicted_Tread < ThresholdTread else False
            json_dump = json.dumps({'Predicted_tread':predicted_Tread ,'N_months':N_months,'Predict_kms':float("% .2f"% Predict_kms),'Need_to_replace':need_to_replace}, cls=NumpyEncoder)  
        result_prediction.append({str(val):json_dump})   
    return {
        'statusCode': 200,
        'body': result_prediction,
        'data': "Results"
    }






def get_token_lite(client_id, user, secret_key, secret_id):
    # Define the necessary information for JWT generation and token endpoint
    return jwt.encode(
        {
            "iss": client_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5),
            "jti": str(uuid.uuid4()),
            "aud": "tableau",
            "sub": user,
            "scp": ["tableau:views:embed", "tableau:metrics:embed"]
        },
            secret_key,
            algorithm = "HS256",
            headers = {
            'kid': secret_id,
            'iss': client_id
            }
    )



def makeTokenFromCredentials():
    client_id  =   config.CLIENT_ID   
    secret_key =  config.SECRET_KEY   
    secret_id  =  config.SECRET_ID    
    user       =  config.USER        
    return get_token_lite(client_id, user, secret_key, secret_id)



# Helper function to make authenticated API calls
def make_request(method, url, headers=None, payload=None):
    if headers is None:
        headers = {}
    response = requests.request(method, url, headers=headers, json=payload)
    return response






@app.route('/')
def main_methods():
    return "ririk"

def Label_encoding_for_features(df,Feature_column):
    if Feature_column=="Applicant_Type__c":
        df.Applicant_Type__c[df.Applicant_Type__c=="Existing Client-Granted Loan Before"] = 0
        df.Applicant_Type__c[df.Applicant_Type__c=="Existing Client-No Loan Granted Ever"] = 1
        df.Applicant_Type__c[df.Applicant_Type__c=="New Client-New Loan"] = 2
    elif Feature_column=="Loan_Amount__c":
        df['Loan_Amount__c'] = df['Loan_Amount__c'].str[1:]
        df['Loan_Amount__c'] = df['Loan_Amount__c'].fillna(0).astype('int64')
    else:
        pass
    return df

def Feature_with_null_values(df):
    features_with_na=[features for features in df.columns if df[features].isnull().sum()>1]
    if len(features_with_na)<1:
        print("There were no features with Null values")
    else:
        print("The Feature with Null values and their percentages are : ")
        
    ## 2- step print the feature name and the percentage of missing values
    for feature in features_with_na:
        print(feature, np.round(df[feature].isnull().mean(), 4),  ' % missing values')

def predicted_results(df):
    df.predicted_result[df.predicted_result==0] = "Green"
    df.predicted_result[df.predicted_result==1] = "Orange"
    df.predicted_result[df.predicted_result==2] = "Red"
    return df

def get_results(df):
    Feature_with_null_values(df)
    # ?????????????????????????? Imputation Technique
    df = Label_encoding_for_features(df,"Applicant_Type__c")
    df = Label_encoding_for_features(df,"Loan_Amount__c")
    df = df.astype('float')
    new_result_df = pd.DataFrame(columns=df.columns)

    predicted_results_arr = []
    for index, row in df.iterrows():
        new_temp_df = pd.DataFrame(columns=df.columns)
        new_temp_df.loc[len(new_result_df.index)] = row
        if (row['Loan_Amount__c']<2000):
            PKL_MODEL = joblib.load(f'static/PKL_Models/{config.PKL_MODEL_DecisionTreeClassifier}')
        if (row['Loan_Amount__c']>2001 and row['Loan_Amount__c']<5000):
            PKL_MODEL = joblib.load(f'static/PKL_Models/MAAC_Models/{config.PKL_MODEL_MAAC_RandomForest}')
        predicted_result =  PKL_MODEL.predict(new_temp_df)
        df.loc[index] = new_temp_df.loc[len(new_result_df.index)]
        predicted_results_arr.append(predicted_result[0])
        del PKL_MODEL,predicted_result,new_temp_df
    
    df['predicted_result'] =  predicted_results_arr
    df = predicted_results(df)
    return df['predicted_result'].values.tolist()


 # Endpoint to create a new guide
@app.route('/api/QueryAnalysis', methods=["POST"])
def add_guide():
    try:
        df = pd.json_normalize(request.json['body'])    
        new_df_with_id = df['id'].values.tolist()
        try:
            df.drop(['id','Opportunity_Origin__c', 'DNB_Scoring_Rate__c', 'Current_Balance__c', 'Opp_number__C', 
                'DP_Budget_Management_Services__c','DP_Monthly_rent_amount_236__c', 'Courts_and_Fines_providers__c',
                'Income_source_is_a_government_benefit__c', 'DP_Primary_income_last_pay_date__c',
                'Primary_regular_benefit_last_pay_date__c', 'Multiple_Lenders_Hardship__c', 'DP_Monthly_rent_amount_236__c',  
                'DP_Monthly_rent_amount_236__c', 'loan_dishonours__c', 'Primary_regular_benefit_monthly_amount__c', 
                'Courts_and_Fines_transactions__c', 'DP_Total_Monthly_Benefit_Income__c', 'Hardship_Indicator_Gambling__c', 
                'SACC_commitments_due_next_month__c', 'Deposits_since_last_SACC_dishonour__c', 'Total_monthly_credits__c',
                'Collection_agency_transactions__c', 'Average_monthly_amount_of_Courts_and_Fin__c', 
                'Deposits_since_last_dishonour__c', 'Bank_Report_Gov_Benefit__c', 'Last_pay_date_for_largest_inc_src_302__c', 
                'Next_pay_date_for_largest_income_source__c'], inplace = True, axis = 1)
        except KeyError:
            print("The columns you are trying to drop had already been droped!")

        new_result_list = []
        new_result_list.append(get_results(df))   #append(df.filter(get_results(df,loaded_model), axis = 1))
        new_df_with_id =   dict(list(zip(new_df_with_id, new_result_list[0])))
        return {
            "Input query":  df.to_dict('dict'),
            "Predicted result":  new_df_with_id,
        }
    except Exception as e:
        return {
            "Status Code":500,
            "Error":"There is some error Occured in code",
            "Exception Occured":str(e)
        }


if __name__ == '__main__':
    app.run(debug=False, port=8001)



# {

# df['StageName'] = df['StageName'].replace(['Closed Lost','Bad Debt Written Off','Bad Debt Pending', 'Bad Debt Watch'], 'Red')
# df['StageName'] = df['StageName'].replace(['Loan Paid'], 'Green')
# df['StageName'] = df['StageName'].replace(['Debt Management', 'Payment Plan', 'Closed Won-Payment Failed', 'Closed Won-Funded'], 'Orange')








    # "id" :                                            "0062x00000DVwMhAAL",
    # "StageName" :                                     "Debt Management",
    # "Opportunity_Origin__c" :                         "QuickCash",
    # "DNB_Scoring_Rate__c" :                           "",
    # "Current_Balance__c" :                            "1085.71",
    # "Applicant_Type__c" :                             "Existing Client-Granted Loan Before",
    # "Opp_number__C" :                                 "5489887",
    # "Multiple_Lenders_Hardship__c" :                  "113.94",
    # "income_as_a_of_DP200_income__c" :                "36.18",
    # "Deposit_spent_on_DOD__c" :                       "73.89",
    # "DP_Monthly_avg_of_SACC_repayments__c" :          "874",
    # "Monthly_ongoing_financial_commitments__c" :      "2482.9198",
    # "DP_Primary_income_frequency__c" :                "2",
    # "DP_Primary_income_last_pay_date__c" :            "30/05/2023",
    # "DP_enders_with_uncleared_dishonours_233__c" :    "0",
    # "Primary_regular_benefit_frequency__c" :          "4",
    # "Last_pay_date_for_largest_inc_src_302__c" :      "30/05/2023",
    # "Largest_income_source_day_of_week__c" :          "2",
    # "Next_pay_date_for_largest_income_source__c" :    "13/06/2023",
    # "Frequency_for_largest_income_source__c" :        "2",
    # "Primary_regular_benefit_last_pay_date__c" :      "24/05/2023",
    # "loan_dishonours__c" :                            "3",
    # "Primary_regular_benefit_monthly_amount__c" :     "1079.8566",
    # "Courts_and_Fines_transactions__c" :              "0",
    # "Total_monthly_income_ongoin_Reg_231__c" :        "4700.2845",
    # "DP_Total_Monthly_Benefit_Income__c" :            "1963",
    # "DP_Dishonours_Across_Primary_Acct_244__c" :      "3",
    # "DP_No_Direct_Debits_On_Primary_Acct_355__c" :    "52",
    # "DP_Budget_Management_Services__c" :              "0",
    # "Hardship_Indicator_Gambling__c" :                "0.83",
    # "DP_Monthly_rent_amount_236__c" :                 "0",
    # "Amount_of_SACC_commitments_due__c" :             "469.74",
    # "Largest_income_Src_Avg_freq__c" :                "938.4142",
    # "Largest_income_Src_last_payment_amt__c" :        "857.5",
    # "Deposits_since_last_SACC_dishonour__c" :         "1",
    # "SACC_commitments_due_next_month__c" :            "8.66",
    # "Total_monthly_credits__c" :                      "5425",
    # "agency_collection_providers__c" :                "1",
    # "Collection_agency_transactions__c" :             "4",
    # "Average_monthly_amount_of_Courts_and_Fin__c" :   "0",
    # "Courts_and_Fines_providers__c" :                 "0",
    # "income_DP200_spend_on_high_risk_merch__c" :      "0.31",
    # "most_recent_loan_has_no_repayments__c" :         "0",
    # "Deposits_since_last_dishonour__c" :              "1",
    # "Income_source_is_other_income_549__c" :          "0",
    # "Bank_Report_Gov_Benefit__c" :                    "0",
    # "Income_source_is_a_government_benefit__c" :      "0",
    # "Summary_Income__c" :                             "0",
    # "Summary_Expenses__c" :                           "400",
    # "Rent_Mortgage__c" :                              "0",
    # "Summary_Total__c" :                              "400",
    # "Loan_Amount__c" :                                "$1000",
    # "Total_Repayment_Amount__c":                      "1360"

































#     "id" :                                                                 "0062x00000EbrjrAAB",
#     "StageName" :                                                          "Closed Lost",
#     "Opportunity_Origin__c" :                                              "MoneySpot",
#     "DNB_Scoring_Rate__c" :                                                "",
#     "Current_Balance__c" :                                                 "",
#     "Applicant_Type__c" :                                                  "Existing Client-Granted Loan Before",
#     "Opp_number__C" :                                                      "6316519",
#     "Multiple_Lenders_Hardship__c" :                                       "35.77",
#     "income_as_a_of_DP200_income__c" :                                     "0",
#     "Deposit_spent_on_DOD__c" :                                            "59.95",
#     "DP_Monthly_avg_of_SACC_repayments__c" :                               "586",
#     "Monthly_ongoing_financial_commitments__c" :                           "1529.4908",
#     "DP_Primary_income_frequency__c" :                                     "1",
#     "DP_Primary_income_last_pay_date__c" :                                 "18/08/2023",
#     "DP_enders_with_uncleared_dishonours_233__c" :                         "0",
#     "Primary_regular_benefit_frequency__c" :                               "0",
#     "Last_pay_date_for_largest_inc_src_302__c" :                           "18/08/2023",
#     "Largest_income_source_day_of_week__c" :                               "4",
#     "Next_pay_date_for_largest_income_source__c" :                         "25/08/2023",
#     "Frequency_for_largest_income_source__c" :                             "1",
#     "Primary_regular_benefit_last_pay_date__c" :                           "",
#     "loan_dishonours__c" :                                                 "0",
#     "Primary_regular_benefit_monthly_amount__c" :                          "0",
#     "Courts_and_Fines_transactions__c" :                                   "0",
#     "Total_monthly_income_ongoin_Reg_231__c" :                             "5353.9583",
#     "DP_Total_Monthly_Benefit_Income__c" :                                 "0",
#     "DP_Dishonours_Across_Primary_Acct_244__c" :                           "0",
#     "DP_No_Direct_Debits_On_Primary_Acct_355__c" :                         "10",
#     "DP_Budget_Management_Services__c" :                                   "0",
#     "Hardship_Indicator_Gambling__c" :                                     "0",
#     "DP_Monthly_rent_amount_236__c" :                                      "0",
#     "Amount_of_SACC_commitments_due__c" :                                  "0",
#     "Largest_income_Src_Avg_freq__c" :                                     "1099.3607",
#     "Largest_income_Src_last_payment_amt__c" :                             "3030.69",
#     "Deposits_since_last_SACC_dishonour__c" :                              "13",
#     "SACC_commitments_due_next_month__c" :                                 "1.15",
#     "Total_monthly_credits__c" :                                           "5405",
#     "agency_collection_providers__c" :                                     "0",
#     "Collection_agency_transactions__c" :                                  "0",
#     "Average_monthly_amount_of_Courts_and_Fin__c" :                        "0",
#     "Courts_and_Fines_providers__c" :                                      "0",
#     "income_DP200_spend_on_high_risk_merch__c" :                           "0",
#     "most_recent_loan_has_no_repayments__c" :                              "0",
#     "Deposits_since_last_dishonour__c" :                                   "13",
#     "Income_source_is_other_income_549__c" :                               "1",
#     "Bank_Report_Gov_Benefit__c" :                                         "FALSE",
#     "Income_source_is_a_government_benefit__c" :                           "0",
#     "Summary_Income__c" :                                                  "3519.64",
#     "Summary_Expenses__c" :                                                "899",
#     "Rent_Mortgage__c" :                                                   "600",
#     "Summary_Total__c" :                                                   "2020.64",
#     "Loan_Amount__c" :                                                     "$4250",
#     "Total_Repayment_Amount__c":                                           "4934.39"
# }

