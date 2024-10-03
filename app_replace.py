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
    Tokens = [makeTokenFromCredentials().decode() for i in range(2)]
    # Tableau Server information
    tableau_server = config.TABLUE_SERVER_URL  #>>>/#/site/mrmt/workbooks/1044061?:origin=card_share_link'
    site_id = config.TABLUE_SITE_ID   # Leave empty for the default site
    contentUrl = config.TABLUE_CONTENT_URL
    months_list = [3,5,7]
    results_of_n_months = []
    for months_index in months_list:
        # Example usage: Retrieve views
        url = f'https://bd8gizd4mg.execute-api.us-east-1.amazonaws.com/prod?months={months_index}'
        headers = {}
        headers['x-api-key'] = 'Dw22gXnwgq3mAg8z5bqA02jJ2fuD9E0r6ulP2iGA'
        response = make_request('GET', url, headers)
        results = response.json()
        if(results['statusCode'] ==200):
            results_body = results['body']
            replacement_true,replacement_flase = [0,0]
            for i in range(len(results_body)):
                for k,v in results_body[i].items():
                    res = json.loads(v)
                    if res['Need_to_replace'] == True:
                        replacement_true += 1
                    elif res['Need_to_replace'] == False:
                        replacement_flase += 1
            # print("Response",results['body'])
            results_of_n_months.append({months_index: {"replacable":replacement_true,"non_replacable":replacement_flase}})
    return render_template('tmass/index.html',results_of_n_months=results_of_n_months,Tokens = Tokens, tableau_server_url=tableau_server, tablue_server_js = config.TABLUE_SERVER_JS ,contentUrl = contentUrl)

@app.route('/twoviewsinone')
def twoviewsinOne_methods():
    Tokens = [makeTokenFromCredentials().decode() for i in range(2)]
    # Tableau Server information
    tableau_server = config.TABLUE_SERVER_URL  #>>>/#/site/mrmt/workbooks/1044061?:origin=card_share_link'
    site_id = config.TABLUE_SITE_ID   # Leave empty for the default site
    contentUrl = config.TABLUE_CONTENT_URL
    months_list = [3,5,7]
    results_of_n_months = []
    for months_index in months_list:
        # Example usage: Retrieve views
        url = f'https://bd8gizd4mg.execute-api.us-east-1.amazonaws.com/prod?months={months_index}'
        headers = {}
        headers['x-api-key'] = 'Dw22gXnwgq3mAg8z5bqA02jJ2fuD9E0r6ulP2iGA'
        response = make_request('GET', url, headers)
        results = response.json()
        if(results['statusCode'] ==200):
            results_body = results['body']
            replacement_true,replacement_flase = [0,0]
            for i in range(len(results_body)):
                for k,v in results_body[i].items():
                    res = json.loads(v)
                    if res['Need_to_replace'] == True:
                        replacement_true += 1
                    elif res['Need_to_replace'] == False:
                        replacement_flase += 1
            # print("Response",results['body'])
            results_of_n_months.append({months_index: {"replacable":replacement_true,"non_replacable":replacement_flase}})
    return render_template('tmass/index2 views.html',results_of_n_months=results_of_n_months,Tokens = Tokens, tableau_server_url=tableau_server, tablue_server_js = config.TABLUE_SERVER_JS ,contentUrl = contentUrl)




@app.route('/request_tablue/')
def python_request_tablue():
    
    tableau_auth = TSC.TableauAuth(config.USER, config.PASSWORD, config.TABLUE_CONTENT_URL)
    server = TSC.Server(config.TABLUE_SERVER_URL)
    server.version = '3.6'

    with server.auth.sign_in(tableau_auth):
        all_views,pagination_item = server.views.get()
        # print([view.content_url for view in all_views])
        # print([view.name for view in all_views])
        count = 0
        selected_views =  all_views[14:]   
        all_Created_images = []
        for view in selected_views:
            print(view.id)
            contenturl = view.content_url
            contenturl = str(contenturl).split("/")
            contenturl = f"{contenturl[0]}/{contenturl[-1]}"
            print("The content URL of the view is ::",contenturl)
            view_item = server.views.get_by_id(view.id)
            all_Created_images.append({"View Id":view.id,"View Title":view.name, "View Image":f"{config.PATH_TO_SAVE_TABLUE_CLOUD_IMAGES}/view_{view.name}.png"})
            server.views.populate_image(view_item)
            png_image = view_item.image
            with open(f'./static/{config.PATH_TO_SAVE_TABLUE_CLOUD_IMAGES}/view_{view.name}.png', 'wb') as f:
                f.write(png_image)
            count += 1

    server.auth.sign_out()
    return render_template('request_tablue/index.html',all_Created_images=all_Created_images)


@app.route('/DashBoard/<dashboard_view_id>/')
def python_check_tablue(dashboard_view_id):
    Tokens = makeTokenFromCredentials().decode()
    tableau_auth = TSC.TableauAuth(config.USER, config.PASSWORD, config.TABLUE_CONTENT_URL)
    server = TSC.Server(config.TABLUE_SERVER_URL)
    TABLUE_CONTENT_URL  = config.TABLUE_CONTENT_URL
    server.version = '3.6'

    all_Created_images = []
    with server.auth.sign_in(tableau_auth):
        
        view = server.views.get_by_id(dashboard_view_id)
        print(view.id)

        # count = 0
        # selected_views =  all_views[14:]  
        contenturl = view.content_url
        contenturl = str(contenturl).split("/")
        contenturl = f"{contenturl[0]}/{contenturl[-1]}"

    server.auth.sign_out()
    print("The Content url is ::: ",contenturl)
    return render_template('request_tablue/dashboard_view.html',contenturl=contenturl,TABLUE_CONTENT_URL=TABLUE_CONTENT_URL ,Tokens=Tokens,tableau_server_url=config.TABLUE_SERVER_URL,tablue_server_js = config.TABLUE_SERVER_JS)

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

def CreateColumnDishonoured(row):
    if row['DP_No_Direct_Debits_On_Primary_Acct_355__c'] == 0:
        return 0
    else:
        return round((row['DP_Dishonours_Across_Primary_Acct_244__c']/row['DP_No_Direct_Debits_On_Primary_Acct_355__c']),2)


def get_results(df):
    Feature_with_null_values(df)
    # ?????????????????????????? Imputation Technique
    df = Label_encoding_for_features(df,"Applicant_Type__c")
    # df = Label_encoding_for_features(df,"Loan_Amount__c")
    df['DP_Dishonours_Across_Primary_Acct_244__c'] = df['DP_Dishonours_Across_Primary_Acct_244__c'].astype('float')
    df['DP_No_Direct_Debits_On_Primary_Acct_355__c'] = df['DP_No_Direct_Debits_On_Primary_Acct_355__c'].astype('float')
    df['dishonoursFound_Created'] = df.apply(CreateColumnDishonoured,axis=1)
    df = df.astype('float')
    new_result_df = pd.DataFrame(columns=df.columns)
    print("This is Ritik", df.columns)

    newOrder = ['Applicant_Type__c' , 'Deposit_spent_on_DOD__c' ,
                'Monthly_ongoing_financial_commitments__c' ,'DP_Primary_income_frequency__c' ,
                'DP_enders_with_uncleared_dishonours_233__c' ,'Largest_income_source_day_of_week__c' ,
                'Frequency_for_largest_income_source__c' ,'Amount_of_SACC_commitments_due__c' ,
                'Largest_income_Src_Avg_freq__c' ,'Largest_income_Src_last_payment_amt__c' ,
                'agency_collection_providers__c' ,'income_DP200_spend_on_high_risk_merch__c' ,
                'most_recent_loan_has_no_repayments__c' ,'Payment_Frequency__c' ,
                'Amount_Requested__c' ,'Dishonours_203__c' ,
                'DP_Days_in_Overdrawn_For_Period__c' ,
                'Dishonours_For_Last_30_Days_DP907__c' ,'Salary_Gov_Allowances_all_types_2117__c' ,
                'DP_Monthly_avg_of_SACC_repayments__c' ,'DP_Total_Monthly_Benefit_Income__c' ,
                'Total_monthly_credits__c' ,'dishonoursFound_Created']
    df = df[newOrder].copy()

    predicted_results_arr = []
    for index, row in df.iterrows():
        new_temp_df = pd.DataFrame(columns=df.columns)
        new_temp_df.loc[len(new_result_df.index)] = row
        # if (row['Amount_Requested__c']<=2000 or row['Amount_Requested__c']>=5001) :
        PKL_MODEL = joblib.load(f'static/PKL_Models/{config.JOBLIB_23_FEATURES_MODEL}')
        # if (row['Amount_Requested__c']>=2001 and row['Amount_Requested__c']<=5000):
        #     PKL_MODEL = joblib.load(f'static/PKL_Models/MAAC_Models/{config.JOBLIB_23_FEATURES_MODEL}')
        predicted_result =  PKL_MODEL.predict(new_temp_df)
        df.loc[index] = new_temp_df.loc[len(new_result_df.index)]
        predicted_results_arr.append(predicted_result[0])
        del PKL_MODEL,predicted_result,new_temp_df
    
    df['predicted_result'] =  predicted_results_arr
    df = predicted_results(df)
    return df['predicted_result'].values.tolist()




@app.route('/api/QueryAnalysis', methods=["POST"])
def add_guide():
    try:
        # Normalize the input JSON into a DataFrame
        df = pd.json_normalize(request.json['body'])
        
        # Extract 'id' values to use later
        ids = df.get('id', []).tolist()
        
        # List of columns to drop
        columns_to_drop = [
            'id', 'Opportunity_Origin__c', 'DNB_Scoring_Rate__c', 'Current_Balance__c', 'Opp_number__C',
            'DP_Budget_Management_Services__c', 'DP_Monthly_rent_amount_236__c', 'Courts_and_Fines_providers__c',
            'Income_source_is_a_government_benefit__c', 'DP_Primary_income_last_pay_date__c', 'Loan_Amount__c', 'Income_source_is_other_income_549__c',
            'Primary_regular_benefit_frequency__c', 'Rent_Mortgage__c', 'Summary_Expenses__c', 'Summary_Income__c', 'Summary_Total__c',
            'Total_Repayment_Amount__c', 'Total_monthly_income_ongoin_Reg_231__c', 'income_as_a_of_DP200_income__c',
            'Primary_regular_benefit_last_pay_date__c', 'Multiple_Lenders_Hardship__c', 'DP_Monthly_rent_amount_236__c',
            'loan_dishonours__c', 'Primary_regular_benefit_monthly_amount__c', 'Courts_and_Fines_transactions__c',
            'Hardship_Indicator_Gambling__c', 'SACC_commitments_due_next_month__c', 'Deposits_since_last_SACC_dishonour__c',
            'Collection_agency_transactions__c', 'Average_monthly_amount_of_Courts_and_Fin__c', 'Deposits_since_last_dishonour__c',
            'Bank_Report_Gov_Benefit__c', 'Last_pay_date_for_largest_inc_src_302__c', 'Next_pay_date_for_largest_income_source__c'
        ]
        # Drop columns only if they exist in the DataFrame
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

        # Generate results from the remaining DataFrame
        result_list = get_results(df)

        # Create a dictionary mapping 'id' values to the corresponding results
        result_dict = dict(zip(ids, result_list))

        # Return the response with input query and predicted result
        return {
            "Input query": df.to_dict('dict'),
            "Predicted result": result_dict
        }

    except Exception as e:
        return {
            "Status Code": 500,
            "Error": "An error occurred in the code",
            "Exception Occurred": str(e)
        }

if __name__ == '__main__':
    app.run(debug=True)



 # 'Applicant_Type__c', 
    #    'Deposit_spent_on_DOD__c',
    #    'Monthly_ongoing_financial_commitments__c',
    #    'DP_Primary_income_frequency__c',
    #    'DP_enders_with_uncleared_dishonours_233__c',
    #    'Largest_income_source_day_of_week__c',
    #    'Frequency_for_largest_income_source__c',
    #    'Amount_of_SACC_commitments_due__c', 
    #    'Largest_income_Src_Avg_freq__c',
    #    'Largest_income_Src_last_payment_amt__c',
    #    'agency_collection_providers__c',
    #    'income_DP200_spend_on_high_risk_merch__c',
    #    'most_recent_loan_has_no_repayments__c', 
    #    'Payment_Frequency__c',
    #    'Amount_Requested__c', 
    #    'Dishonours_203__c',
    #    'DP_Days_in_Overdrawn_For_Period__c',
    #    'Dishonours_For_Last_30_Days_DP907__c',
    #    'Salary_Gov_Allowances_all_types_2117__c',
    #    'DP_Monthly_avg_of_SACC_repayments__c',
    #    'DP_Total_Monthly_Benefit_Income__c', 
    #    'Total_monthly_credits__c',
    #    'dishonoursFound_Created'



# Features In this Order 
'Applicant_Type__c' ,
'Deposit_spent_on_DOD__c' ,
'Monthly_ongoing_financial_commitments__c' ,
'DP_Primary_income_frequency__c' ,
'DP_enders_with_uncleared_dishonours_233__c' ,
'Largest_income_source_day_of_week__c' ,
'Frequency_for_largest_income_source__c' ,
'Amount_of_SACC_commitments_due__c' ,
'Largest_income_Src_Avg_freq__c' ,
'Largest_income_Src_last_payment_amt__c' ,
'agency_collection_providers__c' ,
'income_DP200_spend_on_high_risk_merch__c' ,
'most_recent_loan_has_no_repayments__c' ,
'Payment_Frequency__c' ,
'Amount_Requested__c' ,
'Dishonours_203__c' ,
'DP_Days_in_Overdrawn_For_Period__c' ,
'Dishonours_For_Last_30_Days_DP907__c' ,
'Salary_Gov_Allowances_all_types_2117__c' ,
'DP_Monthly_avg_of_SACC_repayments__c' ,
'DP_Total_Monthly_Benefit_Income__c' ,
'Total_monthly_credits__c' ,
'dishonoursFound_Created'