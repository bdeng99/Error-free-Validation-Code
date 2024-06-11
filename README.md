Read Me:

Download the zipped file and unzip it to a folder.

Download the MNIST data in Matlab data format from
https://lucidar.me/en/matlab/load-mnist-database-of-handwritten-digits-in-matlab/
and save the file to the folder that contains the unzipped Matlab files. 

Open and run Run_Demo_Training.m in Matlab to train an ANN model for MNIST data set to zero error rate. 
For fast demo use small batch sizes around 1000. To train a model with the full size of 60,000 images, 
use model architecture 784-n-10 with n>=100 to save time. Also, in the mfile section titled 
`Stochastic Gradient Descent to prime the search', uncomment  the following lines

% %%%%%%% for full 60000
% nmb_of_searching_per_session=80;
% nmb_of_sessions=4000;
% random_batch_size=1:1000;

before running the script. 

Fully trained models are contained in the data file, Project_1_Model_Parameters.mat. 
To verify that the models achieve the zero error rate, just open and run Run_Demo_Validation.m. 

Change `40' in the Run_Demo_Validation.m file these lines  

W1=h_nodes_40_pr_100_W.one;
W2=h_nodes_40_pr_100_W.two;
b1=h_nodes_40_pr_100_b.one;
b2=h_nodes_40_pr_100_b.two;

to '60', or '80', or '100' to validate the models for different numbers of hidden nodes, 
n, for the 784-n-10 models. 

Bo Deng
Department of Mathematics
Univeristy of Nebraska - Lincoln
Lincoln, NE 68588
bdeng@math.unl.edu
Nov. 2023
