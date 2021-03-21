# nflbigdatabowl
**MISSION:** To be able to predict the number of yards a football team advances on a given rushing play given **49 distinct features** and **500,000 plays** from the NFL. For example, The Patriots have the ball, They're on third down. The quarterback receives the ball and he rushes it. Predict how many yards he advances in the field.

**PROBLEM:** How to make a model that adequately predicts the number of yards a football team will cover on a live game, given game data such as play, position, player, speed, player height, time snap, etc.

**SOLUTION:** Use linear regression. It uses a dependent and independent variable using the best fit line to model the data, regression is the best model as the data is not a binary class. 

**HOW IT WILL WORK?** Linear Regression in combination with cross validation of the data, which comprises of 49 distinct features and 500K plays from the NFL.

**MORE ON HOW IT WILL WORK?:**
* Vectorize our non-integer features to enhance the training of the model.
* Graph features to better understand relationship between features and class. 
* Train data using cross validation and analyze based on cumulative probability and root mean square error. 
*Use graph interpretation to better results by adding more weight to features like rushing player, yard line, and distance till first down.  These features will hold more weight than the others. 
For example:


![image](https://user-images.githubusercontent.com/49461063/111890303-a336c800-89be-11eb-83b2-f113231ab702.png)















![image](https://user-images.githubusercontent.com/49461063/111890311-b2b61100-89be-11eb-84c9-63ab51933260.png) 

















![image](https://user-images.githubusercontent.com/49461063/111890320-c497b400-89be-11eb-98ec-affa6af1aded.png) 


As we can see in the previous graphs these features often lead to a positive yardage gain for majority of the time, where features like team abbreviation would have no effect on the yards gained.

**KEY TAKEAWAYS**
* The NFL Community will benefit by gaining a better understanding of the game and using this as a framework for future analysis. 








