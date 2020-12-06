clear all;
path = "../Yale_Database"; % Full face
m = 195;
n = 231;
num_persons = 15;
img_per_person = 11;
path_crop = "../yaleExpCropped";% Cropped face
m_crop = 50;
n_crop = 52;
%% Eigenfaces Accuracy using full faces on Yale Database
tic;
accuracy_vec = zeros(img_per_person,1);
for i=1:img_per_person
    accuracy_vec(i) = LDA_method(path,(m*n),img_per_person,num_persons,i);
end
disp("Accuracy vector for all types of images are");
disp(accuracy_vec*100);
toc;
%% Fisherfaces with Leaving out method
tic;
accuracy = LDA_Exp2(path,(m*n),img_per_person,num_persons);
disp("Accuracy on full face Yale database is")
disp(accuracy*100);
toc;
%%
tic;
accuracy = LDA_Exp2(path_crop,(m_crop*n_crop),img_per_person,num_persons);
disp("Accuracy on cropped face Yale database is")
disp(accuracy*100);
toc;
