function [accuracy] = LDA_Exp1(set1, set2, set3, set4, set5, setcounter, n, num_person)  
    accuracy = zeros(4,1);
    total1 = zeros(4,1);
    c = num_person;
    m = num_person - 1;
    count = 1;
    N = sum(setcounter(1,:));
    X = zeros(n, N);
    Y = zeros(1, N);
    % dimensionality reduction using PCA 
    for i = 1:1:c
        for j = 1:1:setcounter(1,i)
            X(:, count) = set1(:,j,i);
            Y(1, count) = i;
            count = count + 1;
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
    total = 0;
    for i = 1:1:c
        ind = [(1+total):(total+setcounter(1,i))];
        mu_i(:,i) = sum(X_red(:, ind), 2)/(setcounter(1,i));
        X_cen(:, ind) = X_red(:, ind) - mu_i(:,i);  
        total = total + setcounter(1,i);
    end

    mu = mean(X_red,2);
    Sb = (mu_i-mu) * (mu_i-mu)';
    Sw = X_cen * X_cen';

    [ort, val] = eig(Sb, Sw); % LDA orthonormal basis

    % sort according to eigenvalue
    [val, ind] = sort(diag(val), "descend");
    ort = ort(:, ind);
    X_final = ort(:, 1:m)' * X_red; % I dunno if X_red or X_cen
    
    
    for i = 1:1:num_person
        for j = 1:1:setcounter(2,i)
            img = set2(:,j,i);
            img = img - mu_pca;
            img_red = basis' * img;
            img_final = ort(:, 1:m)' * img_red;
            total1(1) = total1(1) + 1;
            d = inf;
            ind = -1;
            for k = 1:1:N
                if d > sum((img_final - X_final(:, k)).^2)
                    d =  sum((img_final - X_final(:, k)).^2);
                    ind = k;
                end
            end
            if Y(ind) == i 
                accuracy(1) = accuracy(1) + 1;
            end
        end
    end
    for i = 1:1:num_person
        for j = 1:1:setcounter(3,i)
            img = set3(:,j,i);
            img = img - mu_pca;
            img_red = basis' * img;
            img_final = ort(:, 1:m)' * img_red;
            total1(2) = total1(2) + 1;
            d = inf;
            ind = -1;
            for k = 1:1:N
                if d > sum((img_final - X_final(:, k)).^2)
                    d =  sum((img_final - X_final(:, k)).^2);
                    ind = k;
                end
            end
            if Y(ind) == i 
                accuracy(2) = accuracy(2) + 1;
            end
        end
    end
    for i = 1:1:num_person
        for j = 1:1:setcounter(4,i)
            img = set4(:,j,i);
            img = img - mu_pca;
            img_red = basis' * img;
            img_final = ort(:, 1:m)' * img_red;
            total1(3) = total1(3) + 1;
            d = inf;
            ind = -1;
            for k = 1:1:N
                if d > sum((img_final - X_final(:, k)).^2)
                    d =  sum((img_final - X_final(:, k)).^2);
                    ind = k;
                end
            end
            if Y(ind) == i 
                accuracy(3) = accuracy(3) + 1;
            end
        end
    end
    for i = 1:1:num_person
        for j = 1:1:setcounter(5,i)
            img = set5(:,j,i);
            img = img - mu_pca;
            img_red = basis' * img;
            img_final = ort(:, 1:m)' * img_red;
            total1(4) = total1(4) + 1;
            d = inf;
            ind = -1;
            for k = 1:1:N
                if d > sum((img_final - X_final(:, k)).^2)
                    d =  sum((img_final - X_final(:, k)).^2);
                    ind = k;
                end
            end
            if Y(ind) == i 
                accuracy(4) = accuracy(4) + 1;
            end
        end
    end
    accuracy = accuracy./total1;
end
