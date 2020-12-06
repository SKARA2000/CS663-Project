clear all;
m = 168;
n = 192;
num_person = 38;
path = "../CroppedYale";
[set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);
%% Experiment 1 on only subset 1
accuracy = LDA_Exp1(set1,set2,set3,set4,set5,setcounter,(m*n),num_person);
disp("Accuracy as illumination angle increases");
disp(accuracy);
%% Experiment 1 on subsets 1 and 5 and extrapolating
accuracy = LDA_Exp1_2(set1,set2,set3,set4,set5,setcounter,(m*n),num_person);
disp("Accuracy as illumination angle increases");
disp(accuracy);
