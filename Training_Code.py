# This code has been created after conducting exploratory analysis and offline training (Loan Data Analysis - Exploration + Training + Scoring (Python Notebook)).
# Hence, all features are hard-coded.
import pandas as pd, numpy as np, random
import xgboost as xgb
import pickle
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from datetime import date, datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib
inline

##########################user defined parameters
import_path = '/Users/psantanu/Downloads/loan_data.csv'  # PAth of input data
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
train_samp_pct = 0.8  # Train Test Split Ratio
label = 'target_label'  # Name of Target VAriable to be created

## XGBoost Hyparameters for Tuning
# Number of trees
n_estimators = [100, 200, 300, 400, 500]
# fraction of features (randomly selected) that will be used to train each tree.
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Maximum number of levels in tree
max_depth = [3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
# Minimum loss reduction required to make a further partition on a leaf node of the tree.
gamma = [0.1, 0.3, 0.5, 0.8]
# Regularization constant
learning_rate = [0.001, 0.003, 0.005, 0.01]
wts = [1, 5, 7, 10, 12]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'colsample_bytree': colsample_bytree,
               'max_depth': max_depth,
               'gamma': gamma,
               'learning_rate': learning_rate,
               #                'eval_metric': ['auc'],
               'subsample': [0.5, 0.8],
               'objective': ['binary:logistic'],
               #                'warm_start': ['True'],
               'scale_pos_weight': wts
               }


################### All Functions
# Create loan status groups Function
def bucket_loan_status(row):
    if row in ['Default', 'Charged Off']:
        return 'bad_status'
    if row in ['Fully Paid']:
        return 'fully_paid_status'
    elif row in ['Current']:
        return 'current_status'
    else:
        return 'late_status'


# Confusion Matrix Plot Function
def PlotConfusionMatrix(y_test, pred, y_test_legit, y_test_fraud):
    cfn_matrix = confusion_matrix(y_test, pred)
    cfn_norm_matrix = np.array([[1.0 / y_test_legit, 1.0 / y_test_legit], [1.0 / y_test_fraud, 1.0 / y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 2, 1)
    sns.heatmap(cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1, 2, 2)
    sns.heatmap(norm_cfn_matrix, cmap='coolwarm_r', linewidths=0.5, annot=True, ax=ax)

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()

    print('---Classification Report---')
    print(classification_report(y_test, pred))


# Plot ROC Curve and Confusion Matrix
def roc_confusion(train_val_df, test_df, xg_reg):
    #     Model Evaluation: Precision Recall F1 Score
    print(precision_recall_fscore_support(test_df[label], xg_reg.predict(test_df[feat_list])))

    #     Model Evaluation: AUC
    train_auc = (roc_auc_score(train_val_df[label], xg_reg.predict_proba(train_val_df[feat_list])[:, 1]))
    test_auc = (roc_auc_score(test_df[label], xg_reg.predict_proba(test_df[feat_list])[:, 1]))
    print(test_auc)

    #     ROC Curve
    fpr, tpr, thresholds = roc_curve(test_df[label], xg_reg.predict_proba(test_df[feat_list])[:, 1])
    plt.figure(figsize=(15, 6))
    plt.plot(fpr, tpr, label='XGBoost (area = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for US Marketplace')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    #     Confusion Matrix
    pred = xg_reg.predict(test_df[feat_list])
    y_train_no = train_val_df[label].shape[0] - train_val_df[label].sum()
    y_train_yes = train_val_df[label].sum()
    y_test_no = test_df[label].shape[0] - test_df[label].sum()
    y_test_yes = test_df[label].sum()

    PlotConfusionMatrix(test_df[label], pred, y_test_no, y_test_yes)

# Main Function
def main():
    #     Import Data
    input_df = pd.read_csv(import_path, sep=',')
    input_df['earliest_cr_line'] = pd.to_datetime(input_df['earliest_cr_line'])
    input_df['days_to_cr'] = (input_df['earliest_cr_line'].apply(lambda row: datetime.now() - row)).dt.days
    print('Shape of the dataframe >>> ', input_df.shape)

    #     Only filter for records which have loan status populated
    input_df = input_df[~input_df['loan_status'].isnull()]

    #     Combine Certain Loan Status Groups
    input_df['status_group'] = input_df['loan_status'].apply(lambda row: bucket_loan_status(row))

    #     Model Base Data: Only Fully paid loans and loans which have gone bad
    model_df = input_df[input_df['status_group'].isin(['fully_paid_status', 'bad_status'])]

    #     Create Target Label
    model_df[label] = model_df['status_group'].apply(lambda row: 1 if row == 'bad_status' else 0)

    #     One-hot encoding
    model_df = pd.get_dummies(model_df, columns=char_feats)
    model_df.rename(columns={'emp_length_10+ years': 'emp_length_10plus_years',
                             'emp_length_< 1 year': 'emp_length_less_1_year'},
                    inplace=True)

    #     Split into Train and Test
    msk = np.random.rand(model_df.shape[0]) <= train_samp_pct
    train_val_df = model_df[msk]

    test_df = model_df[~msk]

    #     XGBoost: Random Search Hyperparameter Optimization
    xgb_class = xgb.XGBClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    xgb_random = RandomizedSearchCV(estimator=xgb_class, param_distributions=random_grid, n_iter=300,
                                    cv=3, verbose=2, random_state=42, n_jobs=-1, scoring='f1')
    # Fit the random search model
    xgb_random.fit(train_val_df[feat_list], train_val_df[label])

    #     Use best model from hyperparameter tuning
    xg_reg = xgb_random.best_estimator_

    random.seed(124)
    xg_reg.fit(train_val_df[feat_list], train_val_df[label])

    #     Model Evaluation: Precision Recall F1 Score. ROC Curve and Confusion Matrix
    roc_confusion(train_val_df, test_df, xg_reg)

    #     Export the model for future use
    fileObject = open('./Loan_Model_Out.obj', 'wb')
    pickle.dump(xg_reg, fileObject)
    fileObject.close()


if __name__ == '__main__':
    main()