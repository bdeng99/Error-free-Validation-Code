Read Me:

Download the MNIST data in Matlab data format from
https://lucidar.me/en/matlab/load-mnist-database-of-handwritten-digits-in-matlab/
and save the file to the folder that contains the unzipped Matlab files. 


Fully trained models are contained in the data file, Project_1_Model_Parameters.mat. 
To verify that the models achieve the zero error rate, just open and run Run_Demo_Validation.m. 


Change `40' in these lines of the Run_Demo_Validation.m file  

W1=h_nodes_40_pr_100_W.one;

W2=h_nodes_40_pr_100_W.two;

b1=h_nodes_40_pr_100_b.one;

b2=h_nodes_40_pr_100_b.two;

to '20', or '60', or '80', or '100' to validate the models for different numbers of hidden nodes, 
n, of the network architecture, 784-n-10. 

The 784-20-10 model is the only known model which has the fewest parameters to achieve the zero error rate for MNIST.  

Bo Deng

Department of Mathematics

Univeristy of Nebraska - Lincoln

Lincoln, NE 68588

bdeng@math.unl.edu

Nov. 2023
