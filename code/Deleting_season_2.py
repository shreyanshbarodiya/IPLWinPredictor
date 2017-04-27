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

df_match_not_season2 = df_match.ix[df_match.Season_Id != 2]

df_del = df_del.merge(df_match_not_season2[['Match_Id']], how='right', on='Match_Id')
df_player_match = df_player_match.merge(df_match_not_season2[['Match_Id']], how='right', on='Match_Id')
df_match = df_match_not_season2

df_del.to_csv('../input_2/updated/Ball_by_Ball.csv')
df_match.to_csv('../input_2/updated/Match_Updated.csv')
df_player.to_csv('../input_2/updated/Player.csv')
df_player_match.to_csv('../input_2/updated/Player_Match.csv')
df_season.to_csv('../input_2/updated/Season.csv')
df_team.to_csv('../input_2/updated/Team.csv')
