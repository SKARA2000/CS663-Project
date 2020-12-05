clear all;
%% Yale database loading
starti=3;
endi=167;
m = 231;
n = 195;
N = 165;
Xtrain = zeros([m*n,N]);
path = "./Yale_Database/";
image_paths = dir(path);
%%
for i=starti:endi
    Xtrain(:,(i-starti+1)) = im2double(reshape(imread("./Yale_Database/"+image_paths(i).name),[m*n,1]));
end
%% Taking mean difference
mean = sum(X_train,2)/N;
X_train = X_train - mean;
%% Taking SVD and all for eigenvector and eigenvalue calculation
L = X_train'*X_train/(N-1);
[U, S, ~] = svd(L);
V = X_train*U;
%% Normalization of Eigenvactor matrix
normv = sqrt(sum(V.^2,1)); 
V = bsxfun(@rdivide,V,normv);
%% Inintializing recognition rate and the k values to be used
recognition_rate = zeros([17,1]);
K = [1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000];
%% Calculating the recognition rate when the top 3 eigenvectors are included
for l = 1:1:17
    k = K(l);
    Vk = V(:,1:k);
    Alpha_ik = Vk'*X_train;
    for i = 1:1:38
        subpath = path + subfolder(i+2).name;
        images = dir(subpath);
        for j = 41:1:60
            TestImg = im2double(reshape(imread(subpath +"/" +images(j+2).name),[m*n,1]));
            TestImg = TestImg - mean;
            Alpha_p = Vk'*TestImg;
            [~, Identity] = min(sum((Alpha_ik - Alpha_p).^2));
            Identity = ceil(Identity/40);

            if Identity == i
                recognition_rate(l) = recognition_rate(l) + 1;
            end
        end    
    end
end
recognition_rate = recognition_rate*100/(20*38);
%% Plotting the recognition rate versus k
figure
plot(recognition_rate)
title("Plot of the Recognition rate versus the number of Eigenvectors taken on the Yale database when the top 3 eigenvactors are included");
xticks([1:1:17])
xlabel('k');
xticklabels({1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000})
grid on;
%% Calculating the recognition rate when the top 3 eigenvectors are excluded
for l = 1:1:17
    k = K(l);
    Vk = V(:,4:(k+3));
    Alpha_ik = Vk'*X_train;
    for i = 1:1:38
        subpath = path + subfolder(i+2).name;
        images = dir(subpath);
        for j = 41:1:60
            TestImg = im2double(reshape(imread(subpath +"/" +images(j+2).name),[m*n,1]));
            TestImg = TestImg - mean;
            Alpha_p = Vk'*TestImg;
            [~, Identity] = min(sum((Alpha_ik - Alpha_p).^2));
            Identity = ceil(Identity/40);

            if Identity == i
                recognition_rate(l) = recognition_rate(l) + 1;
            end
        end    
    end
end
recognition_rate = recognition_rate*100/(20*38);
%% Plotting the recognition rate versus k
figure
plot(recognition_rate)
title("Plot of the Recognition rate versus the number of Eigenvectors taken on the Yale database when the top 3 eigenvactors are excluded");
xticks([1:1:17])
xlabel('k');
xticklabels({1, 2, 3, 5, 10, 15, 20, 30, 50, 60, 65, 75, 100, 200, 300, 500, 1000})
grid on;