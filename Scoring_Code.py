import pandas as pd, pickle
from datetime import date, datetime

########################## User defined parameters
import_path = '/Users/psantanu/Downloads/loan_data.csv'  # PAth of input data (This path currently contains the training data. Please change the path to point to the data containing only the dataset to be scored)
model_path = '/Users/psantanu/Loan_Model_Out.obj'  # Path of Model
char_feats = ['term', 'emp_length', 'home_ownership', 'purpose']  # Categorical FEatures to be one-hot encoded
feat_list = ['funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'mths_since_last_delinq',
             'open_acc', 'revol_bal', 'total_acc', 'days_to_cr', 'term_ 36 months', 'term_ 60 months',
             'emp_length_1 year', 'emp_length_10plus_years', 'emp_length_2 years', 'emp_length_3 years',
             'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years', 'emp_length_7 years',
             'emp_length_8 years', 'emp_length_9 years', 'emp_length_less_1_year', 'home_ownership_MORTGAGE',
             'home_ownership_OWN', 'home_ownership_RENT', 'purpose_car', 'purpose_credit_card',
             'purpose_debt_consolidation', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
             'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
             'purpose_vacation',
             'purpose_wedding']  # List of all features after one-hot encoding

# Main Function
def main():
    #     Import Data
    input_df = pd.read_csv(import_path, sep=',')
    input_df['earliest_cr_line'] = pd.to_datetime(input_df['earliest_cr_line'])
    input_df['days_to_cr'] = (input_df['earliest_cr_line'].apply(lambda row: datetime.now() - row)).dt.days
    print('Shape of the dataframe >>> ', input_df.shape)

    #     One-hot encoding
    input_df = pd.get_dummies(input_df, columns=char_feats)

    input_df.rename(columns={'emp_length_10+ years': 'emp_length_10plus_years',
                             'emp_length_< 1 year': 'emp_length_less_1_year'},
                    inplace=True)

    #     Load Model
    xg_reg = pickle.load(open(model_path, 'rb'))

    #     Score Data using loaded model
    scored_df = pd.DataFrame(xg_reg.predict_proba(input_df[feat_list])[:, 1])
    scored_df.columns = ['pred_score']

    pred_df = pd.DataFrame(xg_reg.predict(input_df[feat_list]))
    pred_df.columns = ['pred_flag']

    input_df.reset_index(drop=True, inplace=True)

    input_df = pd.concat([input_df, scored_df, pred_df], axis=1)

    #     Export the Scored File to the Output Location
    input_df.to_csv('./loan_data_scored.csv', index=False)


if __name__ == '__main__':
    main()