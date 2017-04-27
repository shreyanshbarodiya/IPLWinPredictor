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


df_del = pd.read_csv("../input_2/Ball_by_Ball.csv")
df_match = pd.read_csv("../input_2/Match.csv")
df_player = pd.read_csv("../input_2/Player.csv")
df_player_match = pd.read_csv("../input_2/Player_Match.csv")
df_season = pd.read_csv("../input_2/Season.csv")
df_team = pd.read_csv("../input_2/Team.csv")

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

##########

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
bow.fillna(0, inplace=True)

bm['cum_run_scored'] = bm.groupby(['Striker_Id'])['Batsman_Scored'].cumsum() - bm['Batsman_Scored']
bm['cum_ball_faced'] = bm.groupby(['Striker_Id'])['Balls_Faced'].cumsum() - bm['Balls_Faced']
bm['cum_four'] = bm.groupby(['Striker_Id'])['Fours'].cumsum() - bm['Fours']
bm['cum_six'] = bm.groupby(['Striker_Id'])['Sixes'].cumsum() - bm['Sixes']
bm['cum_dis'] = bm.groupby(['Striker_Id'])['Dismissed'].cumsum() - bm['Dismissed']
bm['Player_Id'] = bm['Striker_Id']
bm = bm.drop(['Batsman_Scored', 'Balls_Faced', 'Fours', 'Sixes', 'Dismissed', 'Striker_Id'], axis=1)


bow['cum_run_conceded'] = bow.groupby(['Bowler_Id'])['Runs Conceded'].cumsum() - bow['Runs Conceded']
bow['cum_ball_thrown'] = bow.groupby(['Bowler_Id'])['Balls Thrown'].cumsum() - bow['Balls Thrown']
bow['cum_wicket'] = bow.groupby(['Bowler_Id'])['Wickets Taken'].cumsum() - bow['Wickets Taken']
bow['Player_Id'] = bow['Bowler_Id']
bow = bow.drop(['Runs Conceded', 'Balls Thrown', 'Wickets Taken', 'Bowler_Id'], axis=1)

player = pd.merge(bm, bow)
print(player.sort_values(['Match_Id', 'Player_Id']).head(100))