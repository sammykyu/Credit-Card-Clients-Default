###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, auc

def plot_barchart(data, features):
    """
    Print out bar charts for the features
    
    inputs:
        -- data: array-like input data
        -- features: list of feature's name
    """
    
    # Create plotting figure
    fig = pl.figure(figsize=(16.5, 5))

    for i, feature in enumerate(features):        
        ax = fig.add_subplot(1, 3, i+1)
        # Create outcomes DataFrame
        all_data = data[[feature, 'Defaulted']]
        # 'X5' (Age) numerical features
        if(feature == 'Age'):
            # Divide the range of data into bins and count default rates
            min_value = all_data[feature].min()
            max_value = all_data[feature].max()
            value_range = max_value - min_value

            bins = np.arange(0, all_data[feature].max() + 10, 10)

            # Overlay each bin's survival rates
            nondefault_vals = all_data[all_data['Defaulted'] == 0][feature].reset_index(drop = True)
            default_vals = all_data[all_data['Defaulted'] == 1][feature].reset_index(drop = True)
            ax.hist(nondefault_vals, bins = bins, alpha = 0.6,
                     color = 'green', label = 'Did not default')
            ax.hist(default_vals, bins = bins, alpha = 0.6,
                     color = 'red', label = 'Defaulted')

            # Add legend to plot
            ax.set_xlim(0, bins.max())
            ax.legend(framealpha = 0.8)

        # 'Categorical' features
        else:

            # Set the various categories
            if(feature == 'Gender'):
                values = {1:'Male', 2:'Female'}
            if(feature == 'Education'):
                values = {0:'Unk. 1', 1:'Grad. Sch', 2:'University', 3:'High Sch.', 4:'Others', 5:'Unk. 2', 6:'Unk. 3'}
            if(feature == 'Marital_Status'):
                values = {0:'Unk.', 1:'Married', 2:'Single', 3:'Others'}

            # Create DataFrame containing categories and count of each
            frame = pd.DataFrame(index = np.arange(len(values)), columns=(feature,'Defaulted','NDefaulted'))
            for i, value in enumerate(values.keys()):
                frame.loc[i] = [value, \
                       len(all_data[(all_data['Defaulted'] == 1) & (all_data[feature] == value)]), \
                       len(all_data[(all_data['Defaulted'] == 0) & (all_data[feature] == value)])]

            # Set the width of each bar
            bar_width = 0.4

            # Display each category's default rates
            keys = list(values.keys())
            labels = list(values.values())
            for i in np.arange(len(frame)):
                nondefault_bar = ax.bar(keys[i], frame.loc[i]['NDefaulted'], width = bar_width, color = 'g')
                default_bar = ax.bar(keys[i], frame.loc[i]['Defaulted'], width= bar_width, color= 'r',bottom=frame.loc[i]['NDefaulted'])
                default_rate = "{:.2%}".format(frame.loc[i]['Defaulted']/(frame.loc[i]['Defaulted'] + frame.loc[i]['NDefaulted']))
                ax.text(keys[i]-0.2, frame.loc[i]['NDefaulted'] + frame.loc[i]['Defaulted'], default_rate)

            ax.set_xticks(keys)
            ax.set_xticklabels(labels, rotation=40)           
            ax.legend((nondefault_bar[0], default_bar[0]),('Not Defaulted', 'Defaulted'), framealpha = 0.8)

        # Common attributes for plot formatting
        #ax.set_xlabel(feature)
        ax.set_ylabel('Number of Credit Card Clients')
        ax.set_title('With \'%s\' Feature'%(feature), fontsize = 12)
    
    fig.suptitle("Credit Card Payment Default Statistics", fontsize = 16, y = 1.03)
    fig.show()

    
def distribution(data, features, transformed = False):
    """
    Visualization code for displaying distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (16.5, 10));
    
    # Skewed feature plotting
    for i, feature in enumerate(features):  
        ax = fig.add_subplot(2, 3, i+1)
        ax.hist(data[feature], bins = 'auto', color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Data Features", fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Original Distributions of Continuous Data Features", fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()

    
def plot_roc(y_test, y_pred):
    """
    Plot the Receiver Operating Characteristics (ROC) curve
    
    inputs:
        -- y_test: list of actual response labels
        -- y_pred: list of predicted response labels
    """
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    lw=2
    pl.figure()
    pl.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc))
    pl.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    pl.xlabel('False positive rate')
    pl.ylabel('True positive rate')
    pl.title('Receiver Operating Characteristics (ROC) curve')
    pl.legend(loc="lower right")
    pl.show()
    
    