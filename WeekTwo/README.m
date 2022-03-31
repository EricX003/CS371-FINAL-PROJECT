Final Project for CS371 - Jiali Graham and Eric Xing 

Week Two's Progress (3-28-22 to 4-1-22)
  Currently we have five different files, the Driver_XOR, FC_Layer, Net, ReLU, and Activation Layer file. We also have the basic feed-forward mechanism (FC and Activation Layers) and back propogation (calculation of the Loss and editing weights through gradient loss). In addition, we have been adding comments onto the code to make it more understandable for an outside perspective. 
  
  Driver_XOR: contains XOR test, expected outputs are [0, 1, 1, 0] and so far are sucessful.
  FC_Layer: creates a fully-connected layer and pushes the data to the next layer
  Net: calls all the layers, takes the input, loss function initializiation and creation 
  ReLU: makes sure no negative numebrs pass through 
  Activation Layer: runs in between FC_Layer and can shift the added .5 to a function of the user's choosing
  
Issues
- 'Leaky reLU' causing a NaN result
    Solved: We took the derivative correctly.

We have deviated from our project proposal slightly, by making the ANN more comprehensible compared to other programs before moving onto CNN. Therefore, future plans have altered somewhat. In Week Eleven we will be working to make our code accomodate CNN models.

Eric's Contributions:

Jiali's Contributions:
Written comments, writing overview, writing report


  
