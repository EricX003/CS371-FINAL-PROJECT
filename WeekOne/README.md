# CS371-FINAL-PROJECT
Final Project for CS371 - Jiali Graham and Eric Xing

Week One's Progress (3-21-22 to 2-25-22)
  As of right now, we have four files, the Driver_XOR, FC_Layer, Net, and ReLU file. 
  
  The Driver_XOR file runs the XOR test with an expected output of [0, 1, 1, 0]. The Loss Function and its derivative are also found here under the Mean Square Error. 
  FC_Layer runs the data through the layer and trains it. 
  The ReLU (Rectified Linear Unit) file makes sure that a negative number cannot pass through.
  Net is where the data is inputted, layers and loss function are added. 
  
Issues 
    - Vector container objects cannot hold user-defined objects 
        Solved: Created a cell array (Possibly produced a bug that caused nonstatic methods to perform operations on clones of the target object, which leads to no update being performed on the passed object)
 
  Future Plans
  In the future, we plan to implement activation layeres which will change our added .5 to a different function of the user's choosing. We are still following our timeline given in the proposal. 
  Therefore, moving forward into Week Ten we plan to let our code accommodate CNN models, a basic feed-forward mechanism, back propogation, and a gradient descent for convolutional kernel. 
  
  Eric's Contributions 
  
  
  Jiali's Contributions 
  - Written overview, commenting code (as code is still being edited, no comments are present in Week One)
