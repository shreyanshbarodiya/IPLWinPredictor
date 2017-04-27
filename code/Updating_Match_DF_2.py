import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier

df_del = pd.read_csv("../input_2/Ball_by_Ball.csv")
df_match = pd.read_csv("../input_2/Match_Updated.csv")
df_player = pd.read_csv("../input_2/Player.csv")
df_player_match = pd.read_csv("../input_2/Player_Match.csv")
df_season = pd.read_csv("../input_2/Season.csv")
df_team = pd.read_csv("../input_2/Team.csv")

def get_batting_first_team(row):
	if row['Toss_Decision']=='bat':	
		return row['Toss_Winner_Id']
	else:
		if row["Opponent_Team_Id"]==row['Toss_Winner_Id']:
			return row["Team_Name_Id"]
		else:
			return row['Opponent_Team_Id']

def get_batting_second_team(row):
	if row['Toss_Decision']=='field':	
		return row['Toss_Winner_Id']
	else:
		if row["Opponent_Team_Id"]==row['Toss_Winner_Id']:
			return row["Team_Name_Id"]
		else:
			return row['Opponent_Team_Id']

df_match['Batting_first_team'] = df_match.apply(lambda row: get_batting_first_team(row), axis=1)
df_match['Batting_second_team'] = df_match.apply(lambda row: get_batting_second_team(row), axis=1)

df_match.to_csv('../input_2/Match_Updated.csv')

print(df_match.head())