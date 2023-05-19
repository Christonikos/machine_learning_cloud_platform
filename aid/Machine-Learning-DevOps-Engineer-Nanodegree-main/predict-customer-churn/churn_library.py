''' This is a library of functions used for the Udacity project aimed at finding
customers who are likely to churn named "Predict Customer Churn.'''

# import libraries
# import os
# os.environ['QT_QPA_PLATFORM']='offscreen'
import joblib
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    df['Churn'] = pd.Series(
        np.where(
            df.Attrition_Flag.values == 'Attrited Customer',
            1,
            0),
        df.index)

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('./images/eda/churn_plot.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_plot.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_plot.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig('./images/eda/Total_Trans_Ct_plot.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap_plot.png', bbox_inches='tight')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
            could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        new_lst = []
        grouping = df.groupby(col).mean()[response]

        for val in df[col]:
            new_lst.append(grouping.loc[val])

        naming = col + '_' + response
        df[naming] = new_lst

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    y = df['Churn']
    X = pd.DataFrame()
    df = encoder_helper(df, cat_columns, response)

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def X_scaler(X_train):
    '''
    scales the the features with StandardScaler

    input:
            X_data: pandas dataframe of X values

    output:
             Scaled X_data as a numpy array
    '''

    X_scaler =preprocessing.StandardScaler().fit(X_train)
    X_scaled = X_scaler.transform(X_train)

    return X_scaled


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.6, str('Random Forest Test (below) Random Forest Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    plt.figure()
    plt.rc('figure', figsize=(8, 8))
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.6, str('Logistic Regression Test (below) Logistic Regression Train (above)'), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save plot
    plt.savefig(
        output_pth +
        'feature_importance_plot.png',
        bbox_inches='tight')
    plt.close()

def scaled_to_df(X_scaled):
    '''
    Transforms the scaled X data form an numpy array in a
    pandas dataframe
    input:
            X_scaled_data: numpy array of scaled X values

    output:
            Scaled X_data as a pandas dataframe

    '''
    X_scaled_to_df = pd.DataFrame(data=X_scaled[1:,1:],    # values
    index=X_scaled[1:,0],    # 1st column as index
    columns=X_scaled[1,1:])  # 1st row as the column names

    X_scaled_to_df['new_col'] = range(1, len(X_scaled_to_df) + 1)

    X_scaled_to_df.reset_index(inplace=True)

    X_scaled_to_df.drop('new_col', 1, inplace=True)

    X_scaled_to_df.rename(columns={'index': 'Customer_Age',   -1.8113682380719456:'Dependent_count',
         1.8699335965070172: 'Months_on_book', 0.12799720077106438:'Total_Relationship_Count',
        -0.33752655855568725:'Months_Inactive_12_mon', -0.4095410717400574: 'Contacts_Count_12_mon',
        -0.014978924063020588: 'Credit_Limit', -1.426587800403695: 'Total_Revolving_Bal',
         0.11374423698095168: 'Avg_Open_To_Buy', -0.3941940983539414: 'Total_Amt_Chng_Q4_Q1',
        -0.16144253814858084: 'Total_Trans_Amt',   0.07528450617554427: 'Total_Trans_Ct',
        -0.9374654617003583: 'Total_Ct_Chng_Q4_Q1',   -0.9967487116819291: 'Avg_Utilization_Ratio',
         0.9461433114361686: 'Gender_Churn', -0.6673602241616174: 'Education_Level_Churn',
         0.9847791574077642: 'Marital_Status_Churn',    0.5810474931896913: 'Income_Category_Churn',
         0.05879710615053545: 'Card_Category_Churn'},inplace=True)
    return X_scaled_to_df


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # instanciate models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # scale X_train
    X_train_scaled = X_scaler(X_train)

    # scale X_test
    X_test_scaled = X_scaler(X_test)

    # fit models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train_scaled, y_train)
    lrc.fit(X_train_scaled, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train_scaled)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test_scaled)

    y_train_preds_lr = lrc.predict(X_train_scaled)
    y_test_preds_lr = lrc.predict(X_test_scaled)

    # store roc curve with score
    lrc_plot = plot_roc_curve(lrc, X_test_scaled, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test_scaled, y_test, ax=ax)
    lrc_plot.plot(ax=ax)
    plt.savefig('./images/results/roc_curve_plot.png', bbox_inches='tight')
    plt.close()

    # save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store model results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)


    # transforming X_test_scaled into pandas dataframe to plot it
    X_train_scaled_df = scaled_to_df(X_train_scaled)

    # store feature importance plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train_scaled_df,
        './images/results/')


if __name__ == "__main__":

    # import the data
    DATA = import_data("./data/bank_data.csv")

    # exploratory data analysis
    perform_eda(DATA)

    # train models and store results
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA, 'Churn')

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
