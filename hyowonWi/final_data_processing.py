import pandas as pd

def Final_data_processing(type, minute):
    if type == 'Test': data = pd.read_csv("./final_data/Final Test Dataset/" + str(minute) + "mins_test.csv").iloc[:,1:]
    elif type == 'Train': data = pd.read_csv("./final_data/Final Train Dataset/" + str(minute) + "min.csv")
    elif type == 'Validation': data = pd.read_csv("./final_data/Final Validation Dataset/" + str(minute) + "mins_val.csv").iloc[:,1:]

    data = data.drop(['gameId'], axis=1)

    data = data.drop(['redWins'], axis=1)
    data = data.replace({'blueWins': 'Win'}, {'blueWins': 1})
    data = data.replace({'blueWins': 'Fail'}, {'blueWins': 0})

    data = data.replace({'blueFirstTowerLane': "[\'TOP_LANE\']"}, {'blueFirstTowerLane': 1})
    data = data.replace({'blueFirstTowerLane': "[\'MID_LANE\']"}, {'blueFirstTowerLane': 2})
    data = data.replace({'blueFirstTowerLane': "[\'BOT_LANE\']"}, {'blueFirstTowerLane': 3})
    data = data.replace({'blueFirstTowerLane': "[]"}, {'blueFirstTowerLane': 0})

    data = data.replace({'redFirstTowerLane': "[\'TOP_LANE\']"}, {'redFirstTowerLane': 1})
    data = data.replace({'redFirstTowerLane': "[\'MID_LANE\']"}, {'redFirstTowerLane': 2})
    data = data.replace({'redFirstTowerLane': "[\'BOT_LANE\']"}, {'redFirstTowerLane': 3})
    data = data.replace({'redFirstTowerLane': "[]"}, {'redFirstTowerLane': 0})

    data['blueAirDragon'] = 0
    data['blueEarthDragon'] = 0
    data['blueWaterDragon'] = 0
    data['blueFireDragon'] = 0
    data['blueElderDragon'] = 0
    data['redAirDragon'] = 0
    data['redEarthDragon'] = 0
    data['redWaterDragon'] = 0
    data['redFireDragon'] = 0
    data['redElderDragon'] = 0

    split_blue_dragon(data, 'blueDragonType')
    data = data.drop(['blueDragonType'], axis=1)

    split_red_dragon(data, 'redDragonType')
    data = data.drop(['redDragonType'], axis=1)

    data.to_csv("./final_data_processed/Final_" + type + "_Dataset_processed/" + type + "_" + str(minute) + "min.csv")


def split_blue_dragon(df, col):
    for i in df.index:
        val = df.loc[i, col]
        splitted = val[1:-1].split(', ')
        for dragon in splitted:
            if(dragon == "'AIR_DRAGON'"):
                df.loc[i, 'blueAirDragon'] = df.loc[i, 'blueAirDragon'] + 1
            elif(dragon == "'EARTH_DRAGON'"):
                df.loc[i, 'blueEarthDragon'] = df.loc[i, 'blueEarthDragon'] + 1
            elif(dragon == "'WATER_DRAGON'"):
                df.loc[i, 'blueWaterDragon'] = df.loc[i, 'blueWaterDragon'] + 1
            elif(dragon == "'FIRE_DRAGON'"):
                df.loc[i, 'blueFireDragon'] = df.loc[i, 'blueFireDragon'] + 1
            elif(dragon == "'ELDER_DRAGON'"):
                df.loc[i, 'blueElderDragon'] = df.loc[i, 'blueElderDragon'] + 1

def split_red_dragon(df, col):
    for i in df.index:
        val = df.loc[i, col]
        splitted = val[1:-1].split(', ')
        for dragon in splitted:
            if (dragon == "'AIR_DRAGON'"):
                df.loc[i, 'redAirDragon'] = df.loc[i, 'redAirDragon'] + 1
            elif (dragon == "'EARTH_DRAGON'"):
                df.loc[i, 'redEarthDragon'] = df.loc[i, 'redEarthDragon'] + 1
            elif (dragon == "'WATER_DRAGON'"):
                df.loc[i, 'redWaterDragon'] = df.loc[i, 'redWaterDragon'] + 1
            elif (dragon == "'FIRE_DRAGON'"):
                df.loc[i, 'redFireDragon'] = df.loc[i, 'redFireDragon'] + 1
            elif (dragon == "'ELDER_DRAGON'"):
                df.loc[i, 'redElderDragon'] = df.loc[i, 'redElderDragon'] + 1
