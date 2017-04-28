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

df_match['Venue_Id'] = df_match['Venue_Name'].astype('category').cat.codes

# features = ['Venue_Id', 'form_diff']

# target = df_match['home_team_wins']
# data = df_match[features[:]]

# target = target.as_matrix()
# data = data.as_matrix()

# logreg = LogisticRegression()
# classifier = MLPClassifier(hidden_layer_sizes=(12, 2),  random_state=0, max_iter=10000)

# score = cross_val_score(logreg, data, target, cv=5)
# print(df_match.tail(20))

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
# bow.sort_values(['Player_Id', 'Match_Id']).to_csv('bow.csv')

player['Batting_Average'] = player.apply(lambda row: (row['cum_run_scored'] / row['cum_dis']) if row['cum_dis'] > 0 else row['cum_run_scored'], axis=1) 
player['Batting_Strike_Rate'] = (player['cum_run_scored'] / player['cum_ball_faced'])*100
player['Hard_Hitter'] = (player['cum_four']*4 + player['cum_six']*6)/player['cum_ball_faced']

player['Economy_Rate'] = player.apply(lambda row : (row['cum_run_conceded']*6/row['cum_ball_thrown']) if row['cum_ball_thrown'] > 0 else 9, axis=1)
player['Bowling_Strike_Rate'] = player.apply(lambda row: (row['cum_ball_thrown']/row['cum_wicket']) if row['cum_wicket'] > 0 else max(24, row['cum_ball_thrown']), axis=1)
player['Bowling_Average'] = player.apply(lambda row: (row['cum_run_conceded']/row['cum_wicket']) if row['cum_wicket'] > 0 else max(row['Economy_Rate']*4, row['cum_run_conceded']), axis=1)

player.fillna(0, inplace=True)
player.sort_values(['Player_Id', 'Match_Id']).to_csv('XXX.csv')
# player.sort_values(['Player_Id', 'Match_Id']).to_csv('XXX.csv')