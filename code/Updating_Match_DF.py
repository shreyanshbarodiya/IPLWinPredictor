import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier

df_del = pd.read_csv("../input_2/Ball_by_Ball.csv")
df_match = pd.read_csv("../input_2/Match.csv")
df_player = pd.read_csv("../input_2/Player.csv")
df_player_match = pd.read_csv("../input_2/Player_Match.csv")
df_season = pd.read_csv("../input_2/Season.csv")
df_team = pd.read_csv("../input_2/Team.csv")

#calculating the team's season's average 1st innings score
df_temp = pd.merge(df_match[['Match_Id']], df_del[['Match_Id','Innings_Id',  'Total_runs_ball']], left_on='Match_Id', right_on='Match_Id')
df_temp = df_temp.groupby(['Match_Id', 'Innings_Id'])['Total_runs_ball'].sum().reset_index()
df_temp['Inning_Score']=df_temp['Total_runs_ball']
del df_temp['Total_runs_ball']
df_inning_1 = df_temp.ix[df_temp.Innings_Id == 1]
df_inning_2 = df_temp.ix[df_temp.Innings_Id == 2]

df_match = pd.merge(df_match, df_inning_1, left_on='Match_Id', right_on='Match_Id')
df_match['Inning_1_Score']=df_match['Inning_Score']
del df_match['Inning_Score']
del df_match['Innings_Id']

df_match = pd.merge(df_match, df_inning_2, left_on='Match_Id', right_on='Match_Id')
df_match['Inning_2_Score']=df_match['Inning_Score']
del df_match['Inning_Score']
del df_match['Innings_Id']

df_match.to_csv('../input_2/Match_Updated.csv')