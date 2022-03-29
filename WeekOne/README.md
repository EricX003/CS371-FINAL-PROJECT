# CS371-FINAL-PROJECT
Final Project for CS371 - Jiali Graham and Eric Xing

Week One's Progress (3-21-22 to 2-25-22)
  As of right now, we have three files, the Driver_XOR, FC_Layer, Net, and ReLU file. 
  
  The Driver_XOR file runs the XOR test with an expected output of [0, 1, 1, 0]. The Loss Function and its derivative are also found here under the Mean Square Error. 
  FC_Layer runs the data through the layer and trains it. 
  The ReLU (Rectified Linear Unit) file makes sure that a negative number cannot pass through.
  Net is where the data is inputted, layers and loss function are added. 
  
Issues 
    - Object cannot go into a vector 
      >>> Solved: 
      
    - Driver is not recognizing the FC_Layer's functions
      >>> Solved: We have declared each method 'Static' or 'Non-static' in order to prevent confusion from MatLab.
    
  
Future Plans
  In the future, we plan to implement activation layeres which will change our added .5 to a different function of the user's choosing. We are still following our timeline given in the proposal. 
  Therefore, moving forward into Week Ten we plan to let our code accommodate CNN models, a basic feed-forward mechanism, back propogation, and a gradient descent for convolutional kernel. 
  
  Eric's Contributions 
  
  
  Jiali's Contributions 
  - Written overview, commenting code (as code is still being edited, no comments are present in Week One)
