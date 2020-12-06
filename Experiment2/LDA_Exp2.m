function [accuracy] = LDA_Exp2(path, n, img_per_person, num_person)  
    N = (num_person * img_per_person)-1;
    c = num_person;
    m = num_person - 1;
    accuracy = 0;
    images = dir(path);
    lambda = 0;
    % dimensionality reduction using PCA 
    for p = 1:1:c
        for q = 1:1:img_per_person
            X = zeros(n, N);
            Y = zeros(1, N);

            lambda = lambda + 1;
            true_ans = -1;
            count = 1;

            for i = 1:1:c
                for j = 1:1:img_per_person
                    if lambda ~= count 
                        X(:, count) = im2double(reshape(imread(path + "/" + images(count+2).name), [n, 1]));
                        Y(1, count) = i; 
                    else
                        true_ans = i; % class of lambda image 
                    end
                    count = count + 1;
                end
            end
            mu_pca = mean(X, 2);
            X = X - mu_pca;
            mat = X' * X;
            [vec, ~, ~] = svd(mat);

            basis = X * vec(:, 1:N-c);
            normv = sqrt(sum(basis.^2,1)); 
            basis = bsxfun(@rdivide, basis, normv);
            X_red = basis' * X;

            % LDA for nonsingular Sw
            mu_i = zeros(N-c, c);
            X_cen = zeros(size(X_red));
            ind = 1;

            for i = 1:1:c
                flag = false;
                total = img_per_person;
                for j = 1:1:img_per_person
                    if ind == lambda
                        flag = true;
                        total = total - 1;
                    end 
                    if flag == true && j == img_per_person
                        break;
                    end
                    mu_i(:, i) = mu_i(:, i) + X_red(:, ind);
                    ind = ind + 1;
                end
                mu_i(:, i) = mu_i(:, i)/total;
                X_cen(:, (ind-total):(ind-1)) = X_red(:, (ind-total):(ind-1)) - mu_i(i);
            end

            mu = mean(X_red, 2);
            Sb = (mu_i-mu) * (mu_i-mu)';
            Sw = X_cen * X_cen';

            [ort, val] = eig(Sb, Sw); % LDA orthonormal basis

            % sort according to eigenvalue
            [val, ind] = sort(diag(val), "descend");
            ort = ort(:, ind);
            X_final = ort(:, 1:m)' * X_red; % I dunno if X_red or X_cen

            img = im2double(reshape(imread(path + "/" + images(lambda + 2).name), [n, 1]));
            img = img - mu_pca;
            img_red = basis' * img;
            img_final = ort(:, 1:m)' * img_red;

            %% Now find closest neighbor in training set
            [~, ind] = min(sum((img_final - X_final).^2));

            if(Y(1, ind) == true_ans)
                accuracy = accuracy + 1;
            end
        end
    end    
    accuracy = accuracy/(N);
end