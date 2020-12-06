function [accuracy] = LDA_method(path,n,img_per_person,num_person,T)  
    N = num_person*(img_per_person-1);
    c = num_person;
    m = num_person - 1;
    accuracy = 0;
    count = 1;
    X = zeros(n, N);
    Y = zeros(1, N);
    images = dir(path);
    
    % dimensionality reduction using PCA 
    for i = 1:1:c
        for j = 1:1:img_per_person
            if j ~= T
                X(:, count) = im2double(reshape(imread(path + "/" + images(((i-1)*img_per_person) + j + 2).name), [n, 1])); 
                Y(1, count) = i;
                count = count + 1;
            end
        end
    end

    mu_pca = mean(X, 2);
    X = X - mu_pca;
    mat = X' * X;
    [vec, ~, ~] = svd(mat);

    basis = X * vec(:, 1:(N-c));
    normv = sqrt(sum(basis.^2,1)); 
    basis = bsxfun(@rdivide, basis, normv);
    X_red = basis' * X;

    % LDA for nonsingular Sw
    mu_i = zeros((N-c), c);
    X_cen = zeros(size(X_red));
    for i = 1:1:c
        ind = [(1 + (img_per_person-1) * (i-1)):((img_per_person-1) * i)];
        mu_i(:,i) = sum(X_red(:, ind), 2)/(img_per_person-1);
        X_cen(:, ind) = X_red(:, ind) - mu_i(:,i);  
    end

    mu = sum(X_red, 2)/(N);
    Sb = (mu_i-mu) * (mu_i-mu)';
    Sw = X_cen * X_cen';

    [ort, val] = eig(Sb, Sw); % LDA orthonormal basis

    % sort according to eigenvalue
    [val, ind] = sort(diag(val), "descend");
    ort = ort(:, ind);
    X_final = ort(:, 1:m)' * X_red; % I dunno if X_red or X_cen
    
    for i = 1:1:c
        for j = 1:1:img_per_person
            if j == T
                img = im2double(reshape(imread(path + "/" + images(((i-1)*img_per_person) + j + 2).name), [n, 1]));
                img = img - mu_pca;
                img_red = basis' * img;
                img_final = ort(:, 1:m)' * img_red;

                %% Now find closest neighbor in training set
                d = inf;
                ind = -1;
                
                for k = 1:1:N
                    if d > sum((img_final - X_final(:, k)).^2)
                        d =  sum((img_final - X_final(:, k)).^2);
                        ind = k;
                    end
                end

                if Y(ind) == i 
                    accuracy = accuracy + 1;
                end
            end
            count = count+1;
        end
    end
    accuracy = accuracy/num_person;
end