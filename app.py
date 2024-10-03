import pandas as pd
import joblib
import config
from flask import Flask, request, render_template

# creating the flask app
app = Flask(__name__)
app.static_folder = 'static'
# pd.set_option('future.no_silent_downcasting', True)

@app.route('/', methods=["GET"])
def Home():
    return render_template('index.html')


def label_encoding_for_features(df, feature_column):
    # Use replace() for better performance and to avoid SettingWithCopyWarning
    if feature_column == "Applicant_Type__c":
        df['Applicant_Type__c'] = df['Applicant_Type__c'].replace({
            "Existing Client-Granted Loan Before": 0,
            "Existing Client-No Loan Granted Ever": 1,
            "New Client-New Loan": 2
        })
    elif feature_column == "Loan_Amount__c":
        # Use regex to handle currency and missing values efficiently
        df['Loan_Amount__c'] = df['Loan_Amount__c'].replace({'\$': '', ',': ''}, regex=True)
        df['Loan_Amount__c'] = pd.to_numeric(df['Loan_Amount__c'], errors='coerce').fillna(0).astype('int64')
    return df

def feature_with_null_values(df):
    features_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 0]
    if not features_with_na:
        print("There are no features with null values.")
    else:
        print("The features with null values and their percentages are:")
        for feature in features_with_na:
            print(f"{feature}: {df[feature].isnull().mean():.4%} missing values")

def predicted_results(df):
    df['predicted_result'] = df['predicted_result'].replace({
        0: "Green",
        1: "Orange",
        2: "Red"
    })
    return df


def CreateColumnDishonoured(row):
    if row['DP_No_Direct_Debits_On_Primary_Acct_355__c'] == 0:
        return 0
    else:
        return round((row['DP_Dishonours_Across_Primary_Acct_244__c']/row['DP_No_Direct_Debits_On_Primary_Acct_355__c']),2)


def get_results(df):
    feature_with_null_values(df)

    # Apply label encoding
    df = label_encoding_for_features(df, "Applicant_Type__c")

    df['DP_Dishonours_Across_Primary_Acct_244__c'] = df['DP_Dishonours_Across_Primary_Acct_244__c'].astype('float')
    df['DP_No_Direct_Debits_On_Primary_Acct_355__c'] = df['DP_No_Direct_Debits_On_Primary_Acct_355__c'].astype('float')
    df['dishonoursFound_Created'] = df.apply(CreateColumnDishonoured,axis=1)

    # Define new column order
    new_order = [
        'Applicant_Type__c', 'Deposit_spent_on_DOD__c', 'Monthly_ongoing_financial_commitments__c',
        'DP_Primary_income_frequency__c', 'DP_enders_with_uncleared_dishonours_233__c',
        'Largest_income_source_day_of_week__c', 'Frequency_for_largest_income_source__c',
        'Amount_of_SACC_commitments_due__c', 'Largest_income_Src_Avg_freq__c',
        'Largest_income_Src_last_payment_amt__c', 'agency_collection_providers__c',
        'income_DP200_spend_on_high_risk_merch__c', 'most_recent_loan_has_no_repayments__c',
        'Payment_Frequency__c', 'Amount_Requested__c', 'Dishonours_203__c', 'DP_Days_in_Overdrawn_For_Period__c',
        'Dishonours_For_Last_30_Days_DP907__c', 'Salary_Gov_Allowances_all_types_2117__c',
        'DP_Monthly_avg_of_SACC_repayments__c', 'DP_Total_Monthly_Benefit_Income__c', 'Total_monthly_credits__c',
        'dishonoursFound_Created'
    ]
    df = df[new_order].copy()

    predicted_results_arr = []
    
    # Load the model outside the loop for efficiency
    for index, row in df.iterrows():
        # if row['Amount_Requested__c'] <= 2000 or row['Amount_Requested__c'] >= 5001:

        # else:
        #     model_path = f'static/PKL_Models/MAAC_Models/{config.JOBLIB_23_FEATURES_MODEL}'
        
        model_path = f'static/PKL_Models/{config.JOBLIB_23_FEATURES_MODEL}'
        PKL_MODEL = joblib.load(model_path)
        
        predicted_result = PKL_MODEL.predict([row])
        
        predicted_results_arr.append(predicted_result[0])
        del PKL_MODEL

    df['predicted_result'] = predicted_results_arr
    df = predicted_results(df)
    
    return df['predicted_result'].values.tolist()

@app.route('/api/QueryAnalysis', methods=["POST"])
def add_guide():
    # try:
        # Normalize the input JSON into a DataFrame
        df = pd.json_normalize(request.json['body'])
        
        # Extract 'id' values to use later
        ids = df.get('id', []).tolist()
        
        # List of columns to drop
        columns_to_drop = [
            'id', 'Opportunity_Origin__c', 'DNB_Scoring_Rate__c', 'Current_Balance__c', 'Opp_number__C',
            'DP_Budget_Management_Services__c', 'DP_Monthly_rent_amount_236__c', 'Courts_and_Fines_providers__c',
            'Income_source_is_a_government_benefit__c', 'DP_Primary_income_last_pay_date__c', 'Loan_Amount__c',
            'Income_source_is_other_income_549__c', 'Primary_regular_benefit_frequency__c', 'Rent_Mortgage__c',
            'Summary_Expenses__c', 'Summary_Income__c', 'Summary_Total__c', 'Total_Repayment_Amount__c',
            'Total_monthly_income_ongoin_Reg_231__c', 'income_as_a_of_DP200_income__c', 'Primary_regular_benefit_last_pay_date__c',
            'Multiple_Lenders_Hardship__c', 'DP_Monthly_rent_amount_236__c', 'loan_dishonours__c',
            'Primary_regular_benefit_monthly_amount__c', 'Courts_and_Fines_transactions__c', 'Hardship_Indicator_Gambling__c',
            'SACC_commitments_due_next_month__c', 'Deposits_since_last_SACC_dishonour__c', 'Collection_agency_transactions__c',
            'Average_monthly_amount_of_Courts_and_Fin__c', 'Deposits_since_last_dishonour__c', 'Bank_Report_Gov_Benefit__c',
            'Last_pay_date_for_largest_inc_src_302__c', 'Next_pay_date_for_largest_income_source__c'
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

    # except Exception as e:
    #     return {
    #         "Status Code": 500,
    #         "Error": "An error occurred in the code",
    #         "Exception Occurred": str(e)
    #     }



if __name__ == '__main__':
    app.run(debug=True)