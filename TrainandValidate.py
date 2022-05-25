# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
#Data Source
import yfinance as yf
#Data viz
#import plotly.graph_objs as go

#Interval required 5 minutes
market = yf.download(tickers='RWE.DE', period='10y', interval='1d') #gamestop aktie

#delete all rows with Nan values
market.drop(market.index[market['Open'].isnull()], inplace=True) 
#Print data
print(market)

week = []
for i in range(int(len(market)/5)):
    for k in range(5):
        week.append(i+1)
        
for k in range(len(market)%5):
    week.append(i+2)
    
market.index = week

Close_Gap = market['Close'].pct_change()
High_Gap = market['High'].pct_change()
Low_Gap = market['Low'].pct_change() 
Volume_Gap = market['Volume'].pct_change() 
Daily_Change = (market['Close'] - market['Open']) / market['Open']
Outcome_Next_Week_Direction = (market['Volume'].shift(-1) - market['Volume'])
 
market_percentage = pd.DataFrame({'Sequence_ID':[i+1 for i in range(len(market))],
                    'Close_Date':market.index,
                    'Close_Gap':Close_Gap,
                    'High_Gap':High_Gap,
                    'Low_Gap':Low_Gap,
                    'Volume_Gap':Volume_Gap,
                    'Daily_Change':Daily_Change,
                    'Outcome_Next_Week_Direction':Outcome_Next_Week_Direction})

#delete first row
market_percentage = market_percentage.iloc[1: , :] 
#delete all rows with 0 Close Gap
#market_percentage.drop(market_percentage.index[market_percentage['Close_Gap'] == 0], inplace=True) 
#delete all rows with nan outcome next day direction
#market_percentage.drop(market_percentage.index[market_percentage['Outcome_Next_Week_Direction'].isnull()], inplace=True) 

#Binning
# Close_Gap
market_percentage['Close_LMH'] = pd.qcut(market_percentage['Close_Gap'], 3, labels=["L", "M", "H"])

# High
market_percentage['High_LMH'] = pd.qcut(market_percentage['High_Gap'], 3, labels=["L", "M", "H"])

# Low
market_percentage['Low_LMH'] = pd.qcut(market_percentage['Low_Gap'], 3, labels=["L", "M", "H"])

# Volumne
market_percentage['Volume_LMH'] = pd.qcut(market_percentage['Volume_Gap'], 3, labels=["L", "M", "H"])

# Daily_Change
market_percentage['Daily_Change_LMH'] = pd.qcut(market_percentage['Daily_Change'], 3, labels=["L", "M", "H"])


market_binned = pd.DataFrame({'Sequence_ID':[i+1 for i in range(len(market_percentage))],
                    'Close_Date':market_percentage.index,
                    'Close_Gap':market_percentage['Close_LMH'],
                    'High_Gap':market_percentage['High_LMH'],
                    #'Low_Gap':market_percentage['Low_LMH'],
                    'Volume_Gap':market_percentage['Volume_LMH'],
                    'Daily_Change':market_percentage['Daily_Change_LMH'],
                    'Outcome_Next_Week_Direction':market_percentage['Outcome_Next_Week_Direction'],
                    'Pattern': market_percentage['Close_LMH'].astype(str) + market_percentage['Volume_LMH'].astype(str)
                              +market_percentage['Daily_Change_LMH'].astype(str)})


#delete all rows with 0 Close Gap
market_binned.drop(market_binned.index[market_binned['Close_Gap'].isnull()], inplace=True) 
#delete all rows with 0 Volume Gap
market_binned.drop(market_binned.index[market_binned['Volume_Gap'].isnull()], inplace=True) 
#delete all rows with nan outcome next day direction
market_binned.drop(market_binned.index[market_binned['Outcome_Next_Week_Direction'].isnull()], inplace=True) 

market_binned['new week'] = market_binned['Close_Date'].diff()
weekly_outcome = market_binned[market_binned['new week'] != 0]

#compress dataframe from days to weeks
#market_binned_grouped: each row stands for one week with all the patterns for each available day and the resulting outcome next week direction 
market_binned_grouped = market_binned.groupby('Close_Date')['Pattern'].apply(','.join).reset_index()

market_binned_grouped['Outcome_Next_Week_Direction'] = weekly_outcome['Outcome_Next_Week_Direction']

#build training and validation sets
market_training = market_binned_grouped[market_binned_grouped['Close_Date'] <= 200]
market_validation = market_binned_grouped[market_binned_grouped['Close_Date'] > 200]

#keep only big/interesting moves
print('all moves:', len(market_training))
market_training = market_training[abs(market_training['Outcome_Next_Week_Direction']) > 10000]
market_training['Outcome_Next_Week_Direction'] = np.where((market_training['Outcome_Next_Week_Direction'] > 0), 1, 0)
market_validation['Outcome_Next_Week_Direction'] = np.where((market_validation['Outcome_Next_Week_Direction'] > 0), 1, 0)
print('big moves only:', len(market_training)) 

market_training_pos = market_training[market_training['Outcome_Next_Week_Direction']==1][['Close_Date', 'Pattern']]
print(market_training_pos.shape)
market_training_neg = market_training[market_training['Outcome_Next_Week_Direction']==0][['Close_Date', 'Pattern']]
print(market_training_neg.shape)

flat_list = [item.split(',') for item in market_training['Pattern'].values ]
unique_patterns = ','.join(str(r) for v in flat_list for r in v)
unique_patterns = list(set(unique_patterns.split(',')))
len(unique_patterns)

market_training['Outcome_Next_Week_Direction'].head()

# build the markov transition grid
def build_transition_grid(compressed_grid, unique_patterns):
    # build the markov transition grid

    patterns = []
    counts = []
    for from_event in unique_patterns:

        # how many times 
        for to_event in unique_patterns:
            pattern = from_event + ',' + to_event # MMM,MlM

            ids_matches = compressed_grid[compressed_grid['Pattern'].str.contains(pattern)]
            found = 0
            if len(ids_matches) > 0:
                Event_Pattern = '---'.join(ids_matches['Pattern'].values)
                found = Event_Pattern.count(pattern)
            patterns.append(pattern)
            counts.append(found)

    # create to/from grid
    grid_Df = pd.DataFrame({'pairs':patterns, 'counts': counts})

    grid_Df['x'], grid_Df['y'] = grid_Df['pairs'].str.split(',', 1).str
    grid_Df.head()
    
    grid_Df = grid_Df.pivot(index='x', columns='y', values='counts')
    
    print(grid_Df.columns)
    grid_Df.columns= [col for col in grid_Df.columns]
    print(grid_Df.columns)
    grid_Df.index= [col for col in grid_Df.index]
    #del grid_Df.index

    #grid_Df.rowSums(transition_dataframe) 
    grid_Df = grid_Df / grid_Df.sum(1)
    # replace all NaN with zeros
    grid_Df.fillna(0, inplace=True)

    return (grid_Df)

grid_pos = build_transition_grid(market_training_pos, unique_patterns) 
grid_neg = build_transition_grid(market_training_neg, unique_patterns) 

def safe_log(x,y):
   try:
      lg = np.log(x/y)
   except:
      lg = 0
   return lg

# predict on out of sample data
actual = []
predicted = []
for date_id in market_validation['Close_Date'].values:
    patterns = market_validation[market_validation['Close_Date'] == date_id]['Pattern'].values[0].split(',')
    pos = []
    neg = []
    log_odds = []
    
    for id in range(0, len(patterns)-1):
        # get log odds
        # logOdds = log(tp(i,j) / tn(i,j)
        if (patterns[id] in list(grid_pos) and patterns[id+1] in list(grid_pos) and patterns[id] in list(grid_neg) and patterns[id+1] in list(grid_neg)):
                
            numerator = grid_pos[patterns[id]][patterns[id+1]]
            denominator = grid_neg[patterns[id]][patterns[id+1]]
            if (numerator == 0 and denominator == 0):
                log_value =0
            elif (denominator == 0):
                log_value = np.log(numerator / 0.00001)
            elif (numerator == 0):
                log_value = np.log(0.00001 / denominator)
            else:
                log_value = np.log(numerator/denominator)
        else:
            log_value = 0
        
        log_odds.append(log_value)
        
        pos.append(numerator)
        neg.append(denominator)
      
    print('outcome:', market_validation[market_validation['Close_Date']==date_id]['Outcome_Next_Week_Direction'].values[0])
    print(sum(pos)/sum(neg))
    print(sum(log_odds))

    actual.append(market_validation[market_validation['Close_Date']==date_id]['Outcome_Next_Week_Direction'].values[0])
    predicted.append(sum(log_odds))

from sklearn.metrics import confusion_matrix

confusion_matrix(actual, [1 if p > 0 else 0 for p in predicted])

from sklearn.metrics import accuracy_score
score = accuracy_score(actual, [1 if p > 0 else 0 for p in predicted])
print('Accuracy:', round(score * 100,2), '%')



#commit git
# !git init
# !git add README.md
# !git commit -m "first commit"
# !git branch -M main
# !git remote add origin https://github.com/Jogenist/MarkovChain_StockMarket.git
# !git push -u origin main