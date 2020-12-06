%%
clear all;
%%Training on set1,set5 and testing on set2,set3,set4
tic;
%function [accuracy] = Eigenface_Correlation2(path)
    m = 168;
    n = 192;
    num_persons = 38;
    path = "../CroppedYale";
    [set1,set2,set3,set4,set5,setcounter] = Create_Subsets(m,n,path);
    accuracy = zeros([3,1]);
    count = zeros([3,1]);