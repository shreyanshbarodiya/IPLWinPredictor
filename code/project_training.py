from collections import defaultdict
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

NUM_MATCH_HISTORY = 5
team_form = {}
team_form = defaultdict(lambda: deque([]), team_form)

def push_result(queue, result):
	if len(queue) == NUM_MATCH_HISTORY:
		queue.popleft()
	queue.append(result)

def get_form(queue):
	if len(queue) == 0:
		return 0.5
	else:
		return queue.count(1)/len(queue)

def get_form_away_team(row):
	team = row['Opponent_Team_Id']
	queue = team_form[team]
	form = get_form(team_form[team])
	if row['Match_Winner_Id'] == team:
		push_result(queue, 1)
	else:
		push_result(queue, 0)
	return form


def get_form_home_team(row):
	team = row['Team_Name_Id']
	queue = team_form[team]
	form = get_form(team_form[team])
	if row['Match_Winner_Id'] == team:
		push_result(queue, 1)
	else:
		push_result(queue, 0)
	return form


df_del = pd.read_csv("../input_2/updated/Ball_by_Ball.csv")
df_match = pd.read_csv("../input_2/updated/Match_Updated.csv")
df_player = pd.read_csv("../input_2/updated/Player.csv")
df_player_match = pd.read_csv("../input_2/updated/Player_Match.csv")
df_season = pd.read_csv("../input_2/updated/Season.csv")
df_team = pd.read_csv("../input_2/updated/Team.csv")

df_match = df_match[(df_match.Is_DuckWorthLewis == 0) & (df_match.IS_Result == 1)]

df_match['home_team_form'] = df_match.apply(lambda row: get_form_home_team(row), axis=1)
df_match['away_team_form'] = df_match.apply(lambda row: get_form_away_team(row), axis=1)

df_match['form_diff'] = df_match['home_team_form'] - df_match['away_team_form']
df_match['home_team_wins'] = (df_match.Match_Winner_Id == df_match.Team_Name_Id).astype('int')

df_match['City_Id'] = df_match['City_Name'].astype('category').cat.codes

#########################################################################
df_del['Player_dissimal_Id'] = pd.to_numeric(df_del['Player_dissimal_Id'], errors='coerce', downcast='unsigned')
df_del['Batsman_Scored'] = pd.to_numeric(df_del['Batsman_Scored'], errors='coerce', downcast='unsigned')

brs = df_del.groupby(['Match_Id', 'Striker_Id'])['Batsman_Scored'].sum().reset_index()

bbf = df_del[df_del['Extra_Type'] != "wides"].groupby(['Match_Id', 'Striker_Id']).Ball_Id.count().reset_index()
bbf['Balls_Faced'] = bbf['Ball_Id']
bbf = bbf.drop(['Ball_Id'], axis=1)

bfc = df_del[df_del['Batsman_Scored'] == 4].groupby(['Match_Id', 'Striker_Id'])['Ball_Id'].count().reset_index()
bfc['Fours'] = bfc['Ball_Id']
bfc = bfc.drop(['Ball_Id'], axis=1)

bsc = df_del[df_del['Batsman_Scored'] == 6].groupby(['Match_Id', 'Striker_Id'])['Ball_Id'].count().reset_index()
bsc['Sixes'] = bsc['Ball_Id']
bsc = bsc.drop(['Ball_Id'], axis=1)

bo = df_del[df_del['Player_dissimal_Id'] > 0].groupby(['Match_Id','Player_dissimal_Id']).Ball_Id.count().reset_index()
bo['Dismissed'] = bo['Ball_Id']
bo = bo.drop(['Ball_Id'], axis=1)

bm = pd.merge(bbf, brs)
bm = bm.merge(bfc, how='left', on=['Match_Id', 'Striker_Id'])
bm = bm.merge(bsc, how='left', on=['Match_Id', 'Striker_Id'])
bm = bm.merge(bo, how='left', left_on=['Match_Id', 'Striker_Id'], right_on=['Match_Id', 'Player_dissimal_Id'])

bm = bm.drop(['Player_dissimal_Id'], axis=1)
bm.fillna(0, inplace=True)

#########################################################################

rc = df_del[(df_del['Extra_Type'] != "legbyes") & (df_del['Extra_Type'] != "byes")].groupby(['Match_Id', 'Bowler_Id'])['Total_runs_ball'].sum().reset_index()
rc['Runs Conceded'] = rc['Total_runs_ball']
rc = rc.drop(['Total_runs_ball'], axis=1)

bt = df_del[(df_del['Extra_Type'] != "wides") & (df_del['Extra_Type'] != "noballs")].groupby(['Match_Id', 'Bowler_Id']).Ball_Id.count().reset_index()
bt['Balls Thrown'] = bt['Ball_Id']
bt = bt.drop(['Ball_Id'], axis=1)

wt = df_del[(df_del['Player_dissimal_Id'] > 0) & ((df_del['Dissimal_Type'] != "run out") & (df_del['Dissimal_Type'] != "retired hurt"))].groupby(['Match_Id','Bowler_Id']).Ball_Id.count().reset_index()
wt['Wickets Taken'] = wt['Ball_Id']
wt = wt.drop(['Ball_Id'], axis=1)

bow = pd.merge(rc, bt)
bow = bow.merge(wt, how='left', on=['Match_Id', 'Bowler_Id'])
bow['Player_Id'] = bow['Bowler_Id']
bow = bow.drop(['Bowler_Id'], axis=1)
bow.fillna(0, inplace=True)

bm['Player_Id'] = bm['Striker_Id']
bm = bm.drop(['Striker_Id'], axis=1)
bm = df_player_match[['Match_Id', 'Player_Id']].merge(bm, how='left', on=['Match_Id', 'Player_Id'])
bm.fillna(0, inplace=True)

player = bm.merge(bow, how='left', on=['Player_Id', 'Match_Id'])
player.fillna(0, inplace=True)

player['cum_run_scored'] = player.groupby(['Player_Id'])['Batsman_Scored'].cumsum() - player['Batsman_Scored']
player['cum_ball_faced'] = player.groupby(['Player_Id'])['Balls_Faced'].cumsum() - player['Balls_Faced']
player['cum_four'] = player.groupby(['Player_Id'])['Fours'].cumsum() - player['Fours']
player['cum_six'] = player.groupby(['Player_Id'])['Sixes'].cumsum() - player['Sixes']
player['cum_dis'] = player.groupby(['Player_Id'])['Dismissed'].cumsum() - player['Dismissed']

player['cum_run_conceded'] = player.groupby(['Player_Id'])['Runs Conceded'].cumsum() - player['Runs Conceded']
player['cum_ball_thrown'] = player.groupby(['Player_Id'])['Balls Thrown'].cumsum() - player['Balls Thrown']
player['cum_wicket'] = player.groupby(['Player_Id'])['Wickets Taken'].cumsum() - player['Wickets Taken']

player = player.drop(['Batsman_Scored', 'Balls_Faced', 'Fours', 'Sixes', 'Dismissed', 'Runs Conceded', 'Balls Thrown', 'Wickets Taken'], axis=1)

player['Batting_Average'] = player.apply(lambda row: (row['cum_run_scored'] / row['cum_dis']) if row['cum_dis'] > 0 else row['cum_run_scored'], axis=1) 
player['Batting_Strike_Rate'] = (player['cum_run_scored'] / player['cum_ball_faced'])*100
player['Hard_Hitter'] = (player['cum_four']*4 + player['cum_six']*6)/player['cum_ball_faced']

player['Economy_Rate'] = player.apply(lambda row : (row['cum_run_conceded']*6/row['cum_ball_thrown']) if row['cum_ball_thrown'] > 0 else 9, axis=1)
player['Bowling_Strike_Rate'] = player.apply(lambda row: (row['cum_ball_thrown']/row['cum_wicket']) if row['cum_wicket'] > 0 else max(24, row['cum_ball_thrown']), axis=1)
player['Bowling_Average'] = player.apply(lambda row: (row['cum_run_conceded']/row['cum_wicket']) if row['cum_wicket'] > 0 else max(row['Economy_Rate']*4, row['cum_run_conceded']), axis=1)

player.fillna(0, inplace=True)

########################################
#Training
########################################

df_match['target'] = (df_match.Match_Winner_Id == df_match.Batting_first_team).astype('int')
df_training = df_match[['Match_Id','Season_Id', 'City_Name', 'Batting_first_team', 'Batting_second_team', 'target']]


player_stats = ['Batting_Average', 'Batting_Strike_Rate', 'Hard_Hitter', 'Bowling_Average', 'Bowling_Strike_Rate', 'Economy_Rate']
column_names = ['Match_Id','Season_Id', 'City_Name', 'Batting_first_team', 'Batting_second_team', 'target', 'Win_percent_1', 'Win_percent_2', 'Win_Ground_1']

for i in range(132):
	column_names.append('col'+str(i))

num_rows = df_match.shape[0]
new_df = pd.DataFrame(index=np.arange(0, num_rows), columns=column_names)
i=0
for row in df_training.itertuples():
	# new_row = pd.DataFrame('Match_Id', 'City_Name', 'Batting_first_team', 'Batting_second_team')
	match_id = row.Match_Id
	season_id = row.Season_Id
	team1 = row.Batting_first_team
	team2 = row.Batting_second_team
	city = row.City_Name
	target = row.target
	players_1 = df_player_match.ix[(df_player_match.Match_Id == row.Match_Id) & (df_player_match.Team_Id == row.Batting_first_team)]
	players_2 = df_player_match.ix[(df_player_match.Match_Id == row.Match_Id) & (df_player_match.Team_Id == row.Batting_second_team)]
	templist = [match_id, season_id, city, team1, team2, target]
	
	if(season_id>1):
		season_list = df_team_season[(df_team_season.Season_Id == season_id-1) & (df_team_season.Team_Id == team1)]['Win_percent_1'].values.tolist()
		templist.extend(season_list)
		
		season_list = df_team_season[(df_team_season.Season_Id == season_id-1) & (df_team_season.Team_Id == team2)]['Win_percent_2'].values.tolist()
		season_list = -season_list
		templist.extend(season_list)
		
		season_list = df_ground_season[(df_ground_season.Season_Id == season_id-1) & (df_ground_season.City_Name == city)]['Win_percent_1'].values.tolist()
		templist.extend(season_list)
	else:
		templist.append(0.5)
		templist.append(-0.5)
		templist.append(0.5)


	for play in players_1.itertuples():
		player_id = play.Player_Id
		player_stats = player[['Batting_Average', 'Batting_Strike_Rate', 'Hard_Hitter', 'Bowling_Average', 'Bowling_Strike_Rate', 'Economy_Rate']].ix[(player.Match_Id == match_id) & (player.Player_Id == player_id)].values.tolist()
		templist.extend(player_stats[0])

	for play in players_2.itertuples():
		player_id = play.Player_Id
		player_stats = player[['Batting_Average', 'Batting_Strike_Rate', 'Hard_Hitter', 'Bowling_Average', 'Bowling_Strike_Rate', 'Economy_Rate']].ix[(player.Match_Id == match_id) & (player.Player_Id == player_id)].values
		player_stats = -player_stats
		templist.extend(player_stats.tolist()[0])


	if len(templist) > 0:
		new_df.loc[i] = templist
		i+=1

print(new_df.shape)
print(new_df.tail())
new_df.to_csv('Training_1.csv')

# new_df = pd.read_csv('Training_1.csv')
target_df = new_df['target'].astype('int')
training_df = new_df.drop(['Match_Id','Season_Id', 'City_Name', 'Batting_first_team', 'Batting_second_team', 'target'],axis=1)

target = np.array(target_df)
data = np.array(training_df)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

print("classifier  chalu")
classifier = MLPClassifier(hidden_layer_sizes=(139*4, 2), max_iter=10000)
classifier.fit(X_train, y_train)
print("Classifier khatam")

win_prediction = classifier.predict(X_test)
prediction = classifier.predict_proba(X_test)

cnt=0
for i in range(len(win_prediction)):
	if(win_prediction[i]==y_test[i]):
		cnt = cnt + 1

print(cnt/len(win_prediction))

# for i in range(11):

# [['Match_Id', 'Player_Id', 'Team_Id']]