import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier


data_path = "../input/"
score_df = pd.read_csv(data_path+"deliveries.csv")
match_df = pd.read_csv(data_path+"matches.csv")

match_df = match_df.ix[match_df.dl_applied == 0,:]

score_df = pd.merge(score_df, match_df[['id','season', 'winner', 'result', 'dl_applied', 'team1', 'team2']], left_on='match_id', right_on='id')
score_df.player_dismissed.fillna(0, inplace=True)
score_df['player_dismissed'].ix[score_df['player_dismissed'] != 0] = 1
train_df = score_df.groupby(['match_id', 'inning', 'over', 'team1', 'team2', 'batting_team', 'winner'])[['total_runs', 'player_dismissed']].agg(['sum']).reset_index()
train_df.columns = train_df.columns.get_level_values(0)

train_df['innings_wickets'] = train_df.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
train_df['innings_score'] = train_df.groupby(['match_id', 'inning'])['total_runs'].cumsum()

temp_df = train_df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
temp_df['target'] = temp_df['total_runs'] + 1
temp_df = temp_df.ix[temp_df['inning']==1,:]
temp_df['inning'] = 2
temp_df.columns = ['match_id', 'inning', 'target']
train_df = train_df.merge(temp_df, how='left', on = ['match_id', 'inning'])

train_df['score_target'].fillna(-1, inplace=True)

def get_remaining_target(row):
    if row['score_target'] == -1.:
        return -1
    else:
        return row['score_target'] - row['innings_score']

train_df['remaining_target'] = train_df.apply(lambda row: get_remaining_target(row),axis=1)

train_df['run_rate'] = train_df['innings_score'] / train_df['over']

def get_required_rr(row):
    if row['remaining_target'] == -1:
        return -1.
    elif row['over'] == 20:
        return 99
    else:
        return row['remaining_target'] / (20-row['over'])
    
train_df['required_run_rate'] = train_df.apply(lambda row: get_required_rr(row), axis=1)

def get_rr_diff(row):
    if row['inning'] == 1:
        return -1
    else:
        return row['run_rate'] - row['required_run_rate']
    
train_df['runrate_diff'] = train_df.apply(lambda row: get_rr_diff(row), axis=1)
train_df['is_batting_team'] = (train_df['team1'] == train_df['batting_team']).astype('int')
train_df['target'] = (train_df['team1'] == train_df['winner']).astype('int')

train_df.head()

x_cols = ['inning', 'over', 'total_runs', 'player_dismissed', 'innings_wickets', 'innings_score', 'score_target', 'remaining_target', 'run_rate', 'required_run_rate', 'runrate_diff', 'is_batting_team']

val_df = train_df.ix[train_df.match_id == 1,:]
dev_df = train_df.ix[train_df.match_id == 2,:]

dev_X = np.array(dev_df[x_cols[:]])
dev_y = np.array(dev_df['target'])
val_X = np.array(val_df[x_cols[:]])[:-1,:]
val_y = np.array(val_df['target'])[:-1]

classifier = MLPClassifier(hidden_layer_sizes=(12, 2),  random_state=0)
classifier.fit(dev_X, dev_y)
win_prediction = classifier.predict(val_X)

prediction =  classifier.predict_proba(val_X)
print(prediction)


#rr_diff vs prediction
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(1,len(val_X)),val_X[20:,-2], 'b-')
ax1.set_xlabel('Overs')
ax1.set_ylabel('Run rate difference')
ax2 = ax1.twinx()
ax2.plot(range(1,len(val_X)), prediction[20:,0], 'r-')
ax2.set_ylabel('Prediction')


#total_runs vs prediction
#i.e. runs in each over
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(1,len(val_X)),val_X[20:,2], 'b-')
ax1.set_xlabel('Overs')
ax1.set_ylabel('Run scored per over')
ax2 = ax1.twinx()
ax2.plot(range(1,len(val_X)), prediction[20:,0], 'r-')
ax2.set_ylabel('Prediction')
plt.show()

