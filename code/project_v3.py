import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier

df_deliveries = pd.read_csv("../input/deliveries.csv")
df_matches = pd.read_csv("../input/matches.csv")

df_matches = df_matches.loc[df_matches.dl_applied == 0,:]
df_matches = df_matches.loc[df_matches.result == 'normal',:]

df_matches['won_toss'] = (df_matches['team1'] == df_matches['toss_winner']).astype('int')

df_deliveries.player_dismissed.fillna(0, inplace=True)
df_deliveries['player_dismissed'] = df_deliveries.player_dismissed.apply(lambda x : 1 if x!=0 else 0)

df_features = df_deliveries[['match_id', 'inning', 'over', 'total_runs', 'player_dismissed', 'batting_team']]
df_features = df_features.groupby(['match_id', 'inning', 'over', 'batting_team']).sum().reset_index()

df_features['wickets'] = df_features.groupby(['match_id', 'inning'])['player_dismissed'].cumsum()
df_features['score'] = df_features.groupby(['match_id', 'inning'])['total_runs'].cumsum()	

df_temp = df_features.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
df_temp['target'] = df_temp['total_runs'] + 1
df_temp = df_temp.loc[df_temp['inning']==1,:]
df_temp['inning'] = 2
del df_temp['total_runs']

df_ground = df_features.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
df_ground = df_ground.merge(df_matches,  left_on = 'match_id', right_on = 'id')
ground_features  = ['match_id' , 'inning' , 'total_runs', 'city', 'team1', 'winner']
df_ground = df_ground[ground_features[:]]

df_ground['inning_win'] = df_ground.apply(lambda row: 1 if((row['inning']==1 and row['winner']==row['team1']) or (row['inning']==2 and row['winner']!=row['team1'])) else 0, axis=1)
del df_ground['team1']
del df_ground['winner']

df_ground = df_ground.groupby(['inning', 'city'])['total_runs','inning_win'].mean().reset_index()
df_ground['inning_average'] = df_ground['total_runs']
del df_ground['total_runs']

df_features = df_features.merge(df_temp, how='left', on = ['match_id', 'inning'])
df_features['target'].fillna(0, inplace=True)

df_features['rem_target'] = df_features['target']- df_features['score']
df_features['rem_target'] = df_features['rem_target'].apply(lambda x: max(0.0,x))

df_features['rr'] = df_features['score'] / df_features['over']
df_features['rr'].fillna(0, inplace=True)
df_features['rem_overs'] = df_features.over.apply(lambda x: 20-x)
df_features['req_rr'] = df_features['rem_target']/df_features['rem_overs']
df_features['req_rr'].fillna(0, inplace=True)

def get_runrate_diff(row):
    return 0 if row['inning'] == 1 else row['rr'] - row['req_rr']

df_features['rr_diff'] = df_features.apply(lambda row: get_runrate_diff(row), axis=1)

df_features = pd.merge(df_features, df_matches[['id','season', 'winner', 'city', 'won_toss', 'team1', 'team2']], left_on='match_id', right_on='id')

df_features['team1_wins'] = (df_features['team1'] == df_features['winner']).astype('int')


df_features = df_features.merge(df_ground,how='left',  on=['city', 'inning'])

df_features.fillna(0, inplace=True)


features = ['inning', 'over', 'total_runs', 'player_dismissed', 'wickets', 'score', 'rem_target','rr', 'rr_diff', 'inning_win', 'inning_average' ]
# features = ['inning', 'over', 'total_runs', 'player_dismissed', 'wickets', 'score', 'rem_target','rr', 'rr_diff', 'inning_win' ]


df_val = df_features.ix[df_features.match_id == 577 ,:]
# df_val = df_val.ix[df_val.inning==1, :]
# df_val = df_val.ix[ df_val.over==1 , :]
df_train = df_features.ix[df_features.match_id != 577,:]

# plt.figure(figsize=(11,11))
# colormap = plt.cm.viridis_r
# sns.heatmap(df_train[features[:]].corr(), vmax=1.0, cmap=colormap, annot=True)
# plt.show()

train_X = np.array(df_train[features[:]])
train_y = np.array(df_train['team1_wins'])
val_X = np.array(df_val[features[:]])[:-1,:]
val_y = np.array(df_val['team1_wins'])[:-1]

train_X[np.isinf(train_X)] = 1.0e3
val_X[np.isinf(val_X)] = 1.0e3

classifier = MLPClassifier(hidden_layer_sizes=(11, 2),  random_state=0)
classifier.fit(train_X, train_y)
win_prediction = classifier.predict(val_X)

prediction =  classifier.predict_proba(val_X)

# cnt=0
# for i in range(len(win_prediction)):
# 	if(win_prediction[i]!=val_y[i]):
# 		cnt = cnt + 1

# print(cnt/len(win_prediction))

# print(prediction[:,1])

# runs scored in each over vs prediction
fig = plt.figure()
ax1 = fig.add_subplot(111)
bar = ax1.bar(range(1,len(val_X)+1),val_X[:,2])

inn = val_X[:,0]

for b in np.argwhere(inn==2).flatten():
	bar[b].set_color('r')
i = 0
for rect in bar:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., height,
                '\u2022' * int(val_X[i][3]),
                ha='center', va='bottom')
        i+=1
ax1.set_xlabel('Overs')
ax1.set_ylabel('Runs scored in over')
ax2 = ax1.twinx()
ax2.plot(range(1,len(val_X)+1), prediction[:,1], 'b-')
ax2.set_ylabel('Prediction')
plt.show()