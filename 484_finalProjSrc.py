# BEGINNING OF IMPORT STATEMENTS
import numpy as np  # librarires to help with dataframes
import pandas as pd  # libraries to help with dataframes
from itertools import islice  # used to process data
from sklearn.linear_model import LinearRegression  # used to train and predict
from sklearn.model_selection import train_test_split  # used to train data
from kaggle.competitions import nflrush  # used to access train and test data
from sklearn.feature_extraction import DictVectorizer  # used to vectorize data
from math import sqrt  # helped with the calculation of metrics
import math  # helped with calculation of metrics
from sklearn.metrics import mean_squared_error  # used to calculate our scoring metrics
from sklearn import preprocessing  # used to learn about features
from sklearn import metrics  # used to evaluate model
from matplotlib import pyplot as plt  # used to visualize data
from sklearn.preprocessing import OneHotEncoder  # used to vectorize non integer data
from sklearn.preprocessing import LabelEncoder  # Used to Vectorize non integer data
import random  # used to generate random predictions for analysis

env = nflrush.make_env()  # Used to access test data on the kaggle kernel
# END OF IMPORTS

# INITIALIZING AND OPENING FILES
reg = LinearRegression()
myData = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', sep=',', usecols=['DisplayName'])
myTrain = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', sep=',', low_memory=False)
myTrain.PlayDirection.replace(['left', 'right'], [1, 0], inplace=True)
myTrain.Team.replace(['home', 'away'], [1, 0], inplace=True)
myTrain.fillna(0.1, inplace=True)
avgYard = myTrain['Yards'].mean()
le = LabelEncoder()
# In this file processing stage, the first like was to help with knowing what was in
# Our features, the second was to help with opening the actual file
# We then replaced binary data with 1s and 0s because that was the most effective
# way to ensure they were included, we then took care of NaNs so it does not
# cause any future problems.
# END OF FILE PROCESSING


# BEGINNING OF VECTORIZATION STAGES
# In this section o the code we vetorize the labels in our train data
# This is done using Label encoder, other libraries that were tried was dictvectorizer
# As well as OneHotEncoder, but at the end this was the best one to use based on our analysis
# The only thing to note here is that to avoid any erros and stay on th safe side, we
# casted some of the values to a string because that was we wouldnt run into any errors
# because of the variety of types we have and how the library accepts only string data.
myTrain['DisplayName'] = le.fit_transform(myTrain['DisplayName'])
myTrain['GameClock'] = le.fit_transform(myTrain['GameClock'].astype(str))
myTrain['PossessionTeam'] = le.fit_transform(myTrain['PossessionTeam'].astype(str))
myTrain['FieldPosition'] = le.fit_transform(myTrain['FieldPosition'].astype(str))
myTrain['OffenseFormation'] = le.fit_transform(myTrain['OffenseFormation'].astype(str))
myTrain['OffensePersonnel'] = le.fit_transform(myTrain['OffensePersonnel'].astype(str))
myTrain['DefensePersonnel'] = le.fit_transform(myTrain['DefensePersonnel'].astype(str))
myTrain['TimeHandoff'] = le.fit_transform(myTrain['TimeHandoff'].astype(str))
myTrain['TimeSnap'] = le.fit_transform(myTrain['TimeSnap'].astype(str))
myTrain['PlayerHeight'] = le.fit_transform(myTrain['PlayerHeight'].astype(str))
myTrain['PlayerCollegeName'] = le.fit_transform(myTrain['PlayerCollegeName'].astype(str))
myTrain['PlayerBirthDate'] = le.fit_transform(myTrain['PlayerBirthDate'].astype(str))
myTrain['Position'] = le.fit_transform(myTrain['Position'].astype(str))
myTrain['HomeTeamAbbr'] = le.fit_transform(myTrain['HomeTeamAbbr'].astype(str))
myTrain['VisitorTeamAbbr'] = le.fit_transform(myTrain['VisitorTeamAbbr'].astype(str))
myTrain['Stadium'] = le.fit_transform(myTrain['Stadium'].astype(str))
myTrain['Location'] = le.fit_transform(myTrain['Location'].astype(str))
myTrain['StadiumType'] = le.fit_transform(myTrain['StadiumType'].astype(str))
myTrain['Turf'] = le.fit_transform(myTrain['Turf'].astype(str))
myTrain['GameWeather'] = le.fit_transform(myTrain['GameWeather'].astype(str))
myTrain['WindDirection'] = le.fit_transform(myTrain['WindDirection'].astype(str))
myTrain['WindSpeed'] = le.fit_transform(myTrain['WindSpeed'].astype(str))
# END OF VECTORIZATION SECTION

# Here all we are doing is shaping are data and taking a look
# at what we have after everything has been processed
# we also prepare our data for some cross validation by droping the yard column
# which in this case is our class
myTrain.shape
myTrain.describe()

X = myTrain.drop('Yards', axis=1)
y = myTrain['Yards']

# Here we actually start our cross validation we decided to go with a 30/70 split here
# because that gave us the best score when we measured it with RMSE and R2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

rms = sqrt(mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
accu = metrics.r2_score(y_test, y_pred)

randList = []
myRand = 0;
# Here we begin our random yardage prediction to help with the analyzation of our result
# This is done by using the random library provided by python
for i in range(len(X_test)):
    myRand = random.uniform(-99.0, avgYard)
    randList.append(myRand)

# Here we score our metrics to compare to the trained model above.
rms2 = sqrt(mean_squared_error(y_test, randList))
accu2 = metrics.r2_score(y_test, randList)


###########################################
#####BEGINNING OF SOME HELPER FUNCTION#####
####THESE WERE USED IN PYCHARM WHICH ######
###WE CAN ACTUALLY WRITE IT TO A FILE######
# The Data proc function was used to write our confidence values to
# our submission file since the submission can only be created using
# specific data if not the kernel environment will not work
def datProc():
    with open("subfile.txt", "r") as myFile:
        with open("finalKaggleResult.txt", "a") as myFinal:
            for str in myFile:
                str = str.split(' ')
                for itr in str:
                    myFile.write(itr)


# The maDat fucntion here is what actually studies some trends in our
# data and predicts the confidence based on the averages
# and how off a prediction might be
# This is divided into high low and medium confidence values
def maDat():
    with open("finalSub.csv", "w") as mySub:
        mySub.write("Confidence")
        mySub.write("\n")
        with open("submissionEdit.txt", "r") as myFile:
            max = 0
            tot = 0
            avgYard = 4.2123343834966125
            lowConf = avgYard
            highConf = avgYard
            medConf = avgYard
            for line in islice(myFile, 1, 3439):
                myStr = line.split(",")
                for itr in myStr[1:]:
                    tot = tot + int(itr)
                if tot < lowConf:
                    mySub.write("LOW")
                    mySub.write("\n")
                    tot = 0
                if tot > highConf:
                    mySub.write("HIGH")
                    mySub.write("\n")
                    tot = 0
                else:
                    mySub.write("MED")
                    mySub.write("\n")
                    tot = 0


############################################
#######END OF HELPER FUCNTION SECTION#######
############################################
############################################


# Here is the part of our code where we deal with the test data
# Here we need this make_my_predictions function in order for
# the enviroment to work, In here all we are doing is essentially,
# processing the test data like we did above, in the test, with
# the vectorization, and calling our predict function we trained above.

def make_my_predictions(test_dfs, sample_prediction_dfs):
    test_dfs.PlayDirection.replace(['left', 'right'], [1, 0], inplace=True)
    test_dfs.Team.replace(['home', 'away'], [1, 0], inplace=True)
    test_dfs.fillna(0.1, inplace=True)

    test_dfs['DisplayName'] = le.fit_transform(test_dfs['DisplayName'])
    test_dfs['GameClock'] = le.fit_transform(test_dfs['GameClock'].astype(str))
    test_dfs['PossessionTeam'] = le.fit_transform(test_dfs['PossessionTeam'].astype(str))
    test_dfs['FieldPosition'] = le.fit_transform(test_dfs['FieldPosition'].astype(str))
    test_dfs['OffenseFormation'] = le.fit_transform(test_dfs['OffenseFormation'].astype(str))
    test_dfs['OffensePersonnel'] = le.fit_transform(test_dfs['OffensePersonnel'].astype(str))
    test_dfs['DefensePersonnel'] = le.fit_transform(test_dfs['DefensePersonnel'].astype(str))
    test_dfs['TimeHandoff'] = le.fit_transform(test_dfs['TimeHandoff'].astype(str))
    test_dfs['TimeSnap'] = le.fit_transform(test_dfs['TimeSnap'].astype(str))
    test_dfs['PlayerHeight'] = le.fit_transform(test_dfs['PlayerHeight'].astype(str))
    test_dfs['PlayerCollegeName'] = le.fit_transform(test_dfs['PlayerCollegeName'].astype(str))
    test_dfs['PlayerBirthDate'] = le.fit_transform(test_dfs['PlayerBirthDate'].astype(str))
    test_dfs['Position'] = le.fit_transform(test_dfs['Position'].astype(str))
    test_dfs['HomeTeamAbbr'] = le.fit_transform(test_dfs['HomeTeamAbbr'].astype(str))
    test_dfs['VisitorTeamAbbr'] = le.fit_transform(test_dfs['VisitorTeamAbbr'].astype(str))
    test_dfs['Stadium'] = le.fit_transform(test_dfs['Stadium'].astype(str))
    test_dfs['Location'] = le.fit_transform(test_dfs['Location'].astype(str))
    test_dfs['StadiumType'] = le.fit_transform(test_dfs['StadiumType'].astype(str))
    test_dfs['Turf'] = le.fit_transform(test_dfs['Turf'].astype(str))
    test_dfs['GameWeather'] = le.fit_transform(test_dfs['GameWeather'].astype(str))
    test_dfs['WindSpeed'] = le.fit_transform(test_dfs['WindSpeed'].astype(str))
    test_dfs['WindDirection'] = le.fit_transform(test_dfs['WindDirection'].astype(str))

    # In here we also have a set of code to create a dataframe so we can pass it in
    # (more on this later).
    # The picece of code here, essentially just creates 199 columns of the yardage gained or lost
    # just like the kernel has so we can predict and make our own dataframe
    mylist = [[], []]
    myRes = reg.predict(test_dfs)

    ctr = 0
    for i in reversed(range(100)):
        mylist[0].append("Yards-" + str(i))
        if (ctr == 98):
            break
        ctr += 1
    for i in range(0, 100):
        mylist[0].append("Yards" + str(i))
    mylist[1] = [0] * 199
    start = 99
    for j in range(len(myRes)):
        n = math.ceil(myRes[j])
        if (n < 0):
            for i in range(start, start + n, -1):
                mylist[1][i] = 1
        else:
            for i in range(start, start + n):
                mylist[1][i] = 1
    myDFrame = pd.DataFrame([mylist[1]], columns=tuple(mylist[0]))

    return myDFrame

    # This is explained in more detail in our report


# Here we are essentially doing our predictionand writing,
# This function calls the make_my_predictionss fucntions
# which is why we had to write that function ourselves,
# however prior all of this happenning we had to use the
# kernel enviroment given to iterate through each line of the
# test file in order to make some predictions.
def testing():
    itr = 0
    for (test_df, sample_prediction_df) in env.iter_test():
        predictions_df = make_my_predictions(test_df, sample_prediction_df)
        itr += 1
        env.predict(predictions_df)
    env.write_submission_file()


def main():
    testing()


if __name__ == '__main__':
    main()