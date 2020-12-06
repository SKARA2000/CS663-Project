function [set1, set2, set3, set4, set5, setcounter] = Create_Subsets(m,n,path)
    %m = 168;
    %n = 192;
    %path = "../CroppedYale";
    subfolder = dir(path);
    set1 = zeros(m*n,0,38);
    set2 = zeros(m*n,0,38);
    set3 = zeros(m*n,0,38);
    set4 = zeros(m*n,0,38);
    set5 = zeros(m*n,0,38);
    setcounter = zeros(5,38);

    for i = 1:1:38
        set1counter = 0;
        set2counter = 0;
        set3counter = 0;
        set4counter = 0;
        set5counter = 0; 
        subpath = path + "/" +subfolder(i+2).name;
        image_files = dir(subpath);
        for j = 1:1:size(image_files)-2
            image_name = image_files(j+2).name;
            azimuthal = str2double(image_name(14:16));
            elevation = str2double(image_name(19:20));

            if (azimuthal <= 15 && elevation <=15)
                set1counter = set1counter + 1;
                set1(:,set1counter,i) = im2double(reshape(imread(subpath +"/" +image_name),[m*n,1]));
            end
            if (15 < max(azimuthal,elevation) && max(azimuthal,elevation) <=30)
                set2counter = set2counter + 1;
                set2(:,set2counter,i) = im2double(reshape(imread(subpath +"/" +image_name),[m*n,1]));
            end
            if (30 < max(azimuthal,elevation) && max(azimuthal,elevation) <=45)
                set3counter = set3counter + 1;
                set3(:,set3counter,i) = im2double(reshape(imread(subpath +"/" +image_name),[m*n,1]));         
            end
            if (45 < max(azimuthal,elevation) && max(azimuthal,elevation) <=60)
                set4counter = set4counter + 1;
                set4(:,set4counter,i) = im2double(reshape(imread(subpath +"/" +image_name),[m*n,1]));      
            end
            if (60 < max(azimuthal,elevation) && max(azimuthal,elevation) <=75)
                set5counter = set5counter + 1;
                set5(:,set5counter,i) = im2double(reshape(imread(subpath +"/" +image_name),[m*n,1]));           
            end
        end
        setcounter(:,i) = [set1counter set2counter set3counter set4counter set5counter];
    end
end