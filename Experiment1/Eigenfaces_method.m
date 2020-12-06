function [accuracy_top30,accuracy_30] = Eigenfaces_method(set1, set2, set3, set4, set5, setcounter, m, n, K, num_person, flag)
    V_topk = zeros([m*n,K]);
    %correct_recog_topk = zeros(3);
    accuracy=zeros([1,3]);
    if(flag==0)
        V_k = zeros([m*n,K]);
        accuracy1=zeros([1,3]);
    end
    a=0;
    b=0;
    d=0;
    a1=0;
    b1=0;
    d1=0;
    sumcolumn = zeros([1,5]);
    sumcolumn=sum(setcounter');
    x=zeros([1,38]);
    X_train= zeros([m*n,sumcolumn(1)+sumcolumn(5)]);
    c=0;
    for i = 1:1:num_person
        for j = 1:1:setcounter(1,i)
            X_train(:,j+c) = set1(:,j,i);
        end
        for j = 1:1:setcounter(5,i)
            X_train(:,j+setcounter(1,i)+c) = set5(:,j,i);
        end
        c=c+setcounter(1,i)+setcounter(5,i);
        x(i)=c;
    end

    mean = sum(X_train,2)/(sumcolumn(1)+sumcolumn(5));
    X_train = X_train - mean;
    L = X_train'*X_train/(sumcolumn(1)+sumcolumn(5)-1);
    [U,S,~] = svd(L);
    V = X_train*U;
    normv = sqrt(sum(V.^2,1));
    V = bsxfun(@rdivide,V,normv);
    V_topk = V(:,1:K);
    if(flag==0)
        V_k = V(:,4:(K+3));
    end


    for i = 1:1:num_person 
        for j = 1:1:setcounter(2,i)
                Testimg = set2(:,j,i);
                Testimg=Testimg-mean;
                Alpha_ik = V_topk'*X_train;
                Alpha_p = V_topk'*Testimg;
                if(flag==0)
                    Alpha_ik1 = V_k'*X_train;
                    Alpha_p1 = V_k'*Testimg;
                end
                [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
                index=1;
                 while x(index)<Identity
                    index=index+1;
                end
                if(index==i)
                    a=a+1;
                end   
                if(flag==0)
                    [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
                    index=1;
                     while x(index)<Identity
                        index=index+1;
                    end
                    if(index==i)
                        a1=a1+1;
                    end 
                end
        end


        for j = 1:1:setcounter(3,i)
                Testimg = set3(:,j,i);
                Testimg=Testimg-mean;
                Alpha_ik = V_topk'*X_train;
                Alpha_p = V_topk'*Testimg;
                if(flag==0)
                    Alpha_ik1 = V_k'*X_train;
                    Alpha_p1 = V_k'*Testimg;
                end
                [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
                index=1;
                while x(index)<Identity
                    index=index+1;
                end
                if(index==i)
                    b=b+1;
                end 
                if(flag==0)
                    [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
                    index=1;
                     while x(index)<Identity
                        index=index+1;
                    end
                    if(index==i)
                        b1=b1+1;
                    end 
                end                
        end


        for j = 1:1:setcounter(4,i)
                Testimg = set4(:,j,i);
                Testimg=Testimg-mean;
                Alpha_ik = V_topk'*X_train;
                Alpha_p = V_topk'*Testimg;
                if(flag==0)
                    Alpha_ik1 = V_k'*X_train;
                    Alpha_p1 = V_k'*Testimg;
                end
                [~,Identity] = min(sum((Alpha_ik-Alpha_p).^2));
                index=1;
                while x(index)<Identity
                    index=index+1;
                end
                if(index==i)
                   d=d+1;
                end 
                if(flag==0)
                    [~,Identity] = min(sum((Alpha_ik1-Alpha_p1).^2));
                    index=1;
                     while x(index)<Identity
                        index=index+1;
                    end
                    if(index==i)
                        d1=d1+1;
                    end 
                end                
        end

    end
    accuracy(1)=a/sumcolumn(2);
    accuracy(2)=b/sumcolumn(3);
    accuracy(3)=d/sumcolumn(4);
    disp(accuracy);
    accuracy_top30 = accuracy;
    if(flag==0)
        accuracy1(1)=a1/sumcolumn(2);
        accuracy1(2)=b1/sumcolumn(3);
        accuracy1(3)=d1/sumcolumn(4);        
        accuracy_30 = accuracy1;
    else
        accuracy_30 = zeros(3,1);
    end
end