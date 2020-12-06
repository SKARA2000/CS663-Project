%% Clear
clear all;
tic;
%% Function Linear Discriminant Method
%function [accuracy] = LDA(path, T)
    path = "../Yale_Database";
    T = 1;

    n = 195 * 231;
    c = 15;
    img_per_person = 11;
    N = 15 * 10;
    
    accuracy = 0;
    count = 1;
    X = zeros(n, N);
    Y = zeros(1, N);

    images = dir(path);
    
    % dimensionality reduction using PCA 
    for i = 1:1:c
        for j = 1:1:img_per_person
            if j ~= T
                X(:, count) = im2double(reshape(imread(path + "/" + images[count]), [n, 1])); 
                Y(1, count) = i;
                count = count + 1;
            end
        end
    end

    mu_pca = mean(X, 2);
    X = X - mu_pca;
    mat = X' * X;
    [vec, ~, ~] = svd(mat);

    basis = X * vec(:, 1:N-c);
    X_red = basis' * X;

    % LDA for nonsingular Sw
    mu_i = zeros(c, 1);
    
    for i = 1:1:c
        ind = 1 + [(img_per_person-1)*(i-1):(img_per_person-1)*i];
        mu_i(i) = mean(X_red(:, ind), 2);
        X_cen(:, ind) = X_red(:, ind) - mu_i(i);  
    end

    mu = mean(mu_i, 2);
    Sb = (mu_i-mu) * (mu_i-mu)';
    Sw = X_cen * X_cen';

    [ort, ~, ~] = svd(inv(Sw) * Sb); % LDA orthonormal basis
    X_final = ort' * X_red; % I dunno if X_red or X_cen

    for i = 1:1:c
        for j = 1:1:img_per_person
            if j == T
                img = im2double(reshape(imread(path + "/" + images(count)), [n, 1]));
                img = img - mu_pca;
                img_red = basis' * img;
                img_final = ort' * img_red;
                % Now find closest neighbor in training set
            end
        end
    end