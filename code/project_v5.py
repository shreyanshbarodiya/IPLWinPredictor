import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier

df_del = pd.read_csv("../input_2/updated/Ball_by_Ball.csv")
df_match = pd.read_csv("../input_2/updated/Match_Updated.csv")
df_player = pd.read_csv("../input_2/updated/Player.csv")
df_player_match = pd.read_csv("../input_2/updated/Player_Match.csv")
df_season = pd.read_csv("../input_2/updated/Season.csv")
df_team = pd.read_csv("../input_2/updated/Team.csv")

###############################################################################################################
#AVERAGE
#######
####
df_temp = df_match[['Season_Id', 'Batting_first_team', 'Batting_second_team', 'Inning_1_Score', 'Inning_2_Score']]
df_temp1 = df_temp
del df_temp1['Batting_second_team']
del df_temp1['Inning_2_Score']

df_temp1 = df_temp1.groupby(['Season_Id', 'Batting_first_team']).mean().reset_index()
df_average1_season = pd.merge(df_temp1, df_team[['Team_Id']], left_on='Batting_first_team', right_on='Team_Id')
df_average1_season['Inning_1_average'] = df_average1_season['Inning_1_Score']
del df_average1_season['Batting_first_team']
del df_average1_season['Inning_1_Score']

####
df_temp = df_match[['Season_Id', 'Batting_first_team', 'Batting_second_team', 'Inning_1_Score', 'Inning_2_Score']]
df_temp2 = df_temp
del df_temp2['Batting_first_team']
del df_temp2['Inning_1_Score']

df_temp2 = df_temp2.groupby(['Season_Id', 'Batting_second_team']).mean().reset_index()
df_average2_season = pd.merge(df_temp2, df_team[['Team_Id']], left_on='Batting_second_team', right_on='Team_Id')
df_average2_season['Inning_2_average'] = df_average2_season['Inning_2_Score']
del df_average2_season['Batting_second_team']
del df_average2_season['Inning_2_Score']

#WIN PERCENTAGES
#######
####
df_temp = df_match[['Match_Id','Season_Id', 'Batting_first_team']]
df_temp = df_temp.groupby(['Season_Id', 'Batting_first_team'])['Match_Id'].count().reset_index()
df_temp['Team_Id'] = df_temp['Batting_first_team']
df_temp['Total_matches_1'] = df_temp['Match_Id']
del df_temp['Batting_first_team']
del df_temp['Match_Id']
df_temp1 = df_temp

####
df_temp = df_match[['Match_Id','Season_Id', 'Batting_second_team']]
df_temp = df_temp.groupby(['Season_Id', 'Batting_second_team'])['Match_Id'].count().reset_index()
df_temp['Team_Id'] = df_temp['Batting_second_team']
df_temp['Total_matches_2'] = df_temp['Match_Id']
del df_temp['Batting_second_team']
del df_temp['Match_Id']
df_temp2 = df_temp

####
df_temp = df_match[['Match_Id','Season_Id', 'Batting_first_team', 'Match_Winner_Id']]
df_temp = df_temp.ix[df_temp.Batting_first_team == df_temp.Match_Winner_Id]
df_temp = df_temp.groupby(['Season_Id', 'Batting_first_team'])['Match_Id'].count().reset_index()
df_temp['Team_Id'] = df_temp['Batting_first_team']
df_temp['Matches_won_1'] = df_temp['Match_Id']
del df_temp['Batting_first_team']
del df_temp['Match_Id']
df_temp11 = df_temp

####
df_temp = df_match[['Match_Id','Season_Id', 'Batting_second_team','Match_Winner_Id']]
df_temp = df_temp.ix[df_temp.Batting_second_team == df_temp.Match_Winner_Id]
df_temp = df_temp.groupby(['Season_Id', 'Batting_second_team'])['Match_Id'].count().reset_index()
df_temp['Team_Id'] = df_temp['Batting_second_team']
df_temp['Matches_won_2'] = df_temp['Match_Id']
del df_temp['Batting_second_team']
del df_temp['Match_Id']
df_temp22 = df_temp

df_temp = pd.merge(df_temp11, df_temp22) 
#######

df_season_win_percent = pd.merge(df_temp1, df_temp2)
df_season_win_percent = pd.merge(df_season_win_percent, df_temp)
df_season_win_percent['Win_percent_1'] = df_season_win_percent['Matches_won_1']/df_season_win_percent['Total_matches_1']
df_season_win_percent['Win_percent_2'] = df_season_win_percent['Matches_won_2']/df_season_win_percent['Total_matches_2']

df_average_season = pd.merge(df_average1_season, df_average2_season)



###############################################################################################################
#DF_TEAM_SEASON contains ['Season_Id', 'Team_Id', 'Inning_1_average', 'Inning_2_average','Total_matches_1', 'Total_matches_2', 'Matches_won_1', 'Matches_won_2','Win_percent_1', 'Win_percent_2']
df_team_season = pd.merge(df_average_season, df_season_win_percent)



