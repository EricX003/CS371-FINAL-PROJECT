Final Project for CS371 - Jiali Graham and Eric Xing 

Week Two's Progress (3-28-22 to 4-1-22)
  Currently we have five different files, the Driver_XOR, FC_Layer, Net, ReLU, and Activation Layer file.
  
  Driver_XOR: contains XOR test, expected outputs are [0, 1, 1, 0] and so far are sucessful.
  FC_Layer: creates a fully-connected layer and pushes the data to the next layer
  Net: calls all the layers, takes the input, loss function initializiation and creation 
  ReLU: makes sure no negative numebrs pass through 
  Activation Layer: runs in between FC_Layer and can shift the added .5 to a function of the user's choosing
  
Issues

'Leaky reLU' causing a NaN result
  Solved: We took the derivative correctly.


  
