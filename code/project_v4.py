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

df_match = df_match[(df_match.Is_DuckWorthLewis == 0) & (df_match.IS_Result == 1)]

df_match['team1_won'] = df_match[df.Match_Winner_Id == df.Team_Name_Id].groupby('')