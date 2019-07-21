% Ben, 4 Jan, 2018
% Valization of point cloud and 3D bounding box

% reading the point cloud, 
% filetering point cloud dataset
% reading the label
% plotting the 3D bounding box according to label information

% "category", "length", "wdith", "height", "alpha(ration_y)", "x", "y" and "z"
% flipping the point cloud and labels and showing more
function [ ]=main()
    clear all, close all;
    h1 = [];
    label_folder_dir = '/home/ben/Dataset/KITTI/data_object_image_2/training/label_2';
    point_cloud_folder_dir = '/home/ben/Dataset/KITTI/data_object_image_2/training/velodyne';
    img_folder_dir = '/home/ben/Dataset/KITTI/data_object_image_2/training/image_2';
    calib_dir = '/home/ben/Dataset/KITTI/data_object_image_2/training/calib';
    
    assert(exist(label_folder_dir) == 7, 'check the label directory');
    assert(exist(point_cloud_folder_dir) == 7, 'check the point cloud directory');

    all_label = dir(fullfile(label_folder_dir,'*.txt'));
    num_data = length(all_label);

    t_velo_cam = [7.533745e-03, -9.999714e-01, -6.166020e-04 , -4.069766e-03;
                 1.480249e-02, 7.280733e-04,  -9.998902e-01,  -7.631618e-02 ;
                 9.998621e-01, 7.523790e-03,  1.480755e-02,   -2.717806e-01;
                 0, 0, 0, 1 ];
    RO = [ 9.999239e-01,  9.837760e-03, -7.445048e-03, 0;
           -9.869795e-03, 9.999421e-01, -4.278459e-03, 0;
           7.402527e-03,  4.351614e-03, 9.999631e-01, 0;
           0, 0, 0, 1
        ];
    P2 = [ 7.215377e+02,  0.,            6.095593e+02, 4.485728e+01 ;
           0.,            7.215377e+02,  1.728540e+02, 2.163791e-01 ;
           0.,            0.,            1.,         2.745884e-03 ;
           0.,            0.,            0.,         0
        ];
    
    
    X_after_rotation = @(x,y,eta) x*cos(eta) - y*sin(eta) ; % X = x*cos(?) - y*sin(?)
    Y_after_rotation = @(x,y,eta) x*sin(eta)  + y*cos(eta); % Y = x*sin(?) + y*cos(?)
    % 5316 == 5952
    %106test_dir
    test_dir = './r/'
    % 32 33, 115, 135, 139, 143, 153, 158, 212, 218
    for index = 223:num_data
        label_name = all_label(index).name ; 
        clearvars label_data
        label_data = reading_label_data(label_folder_dir, label_name);
        T = Fun_open_calib( all_label(index).name, calib_dir) ;
        
       % plotting point cloud
%             [X2,map2] = imread('./image_02/000000.png');
%             subplot(2,1,1), axis off, imshow(X2)%subimage(X2); 
        point_cloud_name = strrep(label_name, '.txt', '.bin');
        point_cloud = prepare_point_cloud_data(point_cloud_folder_dir ,point_cloud_name);
        delete(h1);
        h1 = figure(1);
        subplot(2,1,1)
        pcshow(point_cloud(:,1:3));

        colormap colorcube
        hold on;
        num_label = size( label_data{1,1} ,1) ;
        point = zeros(8,3);
        bbox_img_car = [];
        bbox_img_pedestrian = [];
        bbox_img_cyclist = [];
        for ii = 1:num_label
            %if  strcmp(label_data{1}{ii}, 'Car') |  strcmp(label_data{1}{ii}, 'Pedestrian') | strcmp(label_data{1}{ii}, 'Cyclist') |  strcmp(label_data{1}{ii}, 'Truck')
            if ~strcmp(label_data{1}{ii}, 'DontCare')
                len = label_data{1,11}(ii);
                wid = label_data{1, 10}(ii);
                hei = label_data{1, 9}(ii);
                alph =  -label_data{1, 15}(ii);  %% clockwise direction
                % the donot care category
                if len == -1
                    len = 4;
                    hei = 2;
                    wid = 2;
                    alph = 0
                end
                
                CTR_PC = (inv(T.Tr_velo_to_cam)*[label_data{1, 12}(ii); label_data{1 , 13}(ii); label_data{1 , 14}(ii); 1])';
                ctr_x = CTR_PC(1);
                ctr_y = CTR_PC(2);
                ctr_z = CTR_PC(3);
                
                % new center and size
                clearvars new_point_cloud, index,
                new_point_cloud = point_cloud(:,1:3) - [ctr_x, ctr_y, ctr_z];
                new_point_cloud(:,1) = X_after_rotation(new_point_cloud(:,2), new_point_cloud(:,1), -alph);
                new_point_cloud(:,2) = Y_after_rotation(new_point_cloud(:,2), new_point_cloud(:,1), -alph);
                %index = and((point_cloud(:,2) < point_cloud(:,1) - 0.27), ( -point_cloud(:,2) < point_cloud(:,1) - 0.27));
                index = (-len/2<new_point_cloud(:,1))&(new_point_cloud(:,1)<len/2)...
                        &(-wid/2<new_point_cloud(:,2))&(new_point_cloud(:,2)<wid/2)...
                        &(0.1<new_point_cloud(:,3))&(new_point_cloud(:,3)<(hei));
                inside_points = new_point_cloud(index,:);
                if size(inside_points, 1) == 0
                    continue
                end
                len = max(inside_points(:,1)) - min(inside_points(:,1));
                x_min = min(inside_points(:,1)); x_max = max(inside_points(:,1));
                %ctr_x = ctr_x - mean(inside_points(:,1));
                wid = max(inside_points(:,2)) - min(inside_points(:,2));
                y_min = min(inside_points(:,2)); y_max=max(inside_points(:,2));
                %ctr_y = ctr_y - mean(inside_points(:,2));
                hei = max(inside_points(:,3)) - min(inside_points(:,3));
             
                z_min = min(inside_points(:,3)); z_max = max(inside_points(:,3));
                %ctr_z = ctr_z - mean(inside_points(:,3));
%                 point(1:4,3) = z_min;
%                 point(5:8,3) = z_max;
%                 point(1,1:2) = [x_min, y_min];
%                 point(2,1:2) = [x_min, y_max];
%                 point(3,1:2) = [x_max, y_max];
%                 point(4,1:2) = [x_max, y_min];
%                 
%                 point(5,1:2) = [x_min, y_min];
%                 point(6,1:2) = [x_min, y_max];
%                 point(7,1:2) = [x_max, y_max];
%                 point(8,1:2) = [x_max, y_min]; 
%                 point(:,1) = X_after_rotation(point(:,2), point(:,1), alph) + ctr_x;
%                 point(:,2) = Y_after_rotation(point(:,2), point(:,1), alph) + ctr_y;
%                 point(:,3) = point(:,3) + ctr_z;
                retangle = [ wid/2,  len/2;
                             wid/2, -len/2;
                             -wid/2, -len/2;
                             -wid/2,  len/2];  %(x, y)
               
                point(1:4,3) = ctr_z ;
                point(5:8,3) = ctr_z + hei;

                point(1,1:2) = [ X_after_rotation( wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( wid/2, len/2, alph) + ctr_y];
                point(2,1:2) = [ X_after_rotation( wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( wid/2, -len/2, alph) + ctr_y];
                point(3,1:2) = [ X_after_rotation( -wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( -wid/2, -len/2, alph) + ctr_y];
                point(4,1:2) = [ X_after_rotation( -wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( -wid/2, len/2, alph) + ctr_y];
                
                point(5,1:2) = [ X_after_rotation( wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( wid/2, len/2, alph) + ctr_y];
                point(6,1:2) = [ X_after_rotation( wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( wid/2, -len/2, alph) + ctr_y];
                point(7,1:2) = [ X_after_rotation( -wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( -wid/2, -len/2, alph) + ctr_y];
                point(8,1:2) = [ X_after_rotation( -wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( -wid/2, len/2, alph) + ctr_y];
                
                points_rotation = ones(size(point,1),4);
                points_rotation(:,1:3) = point;
                px = (T.P2 * T.R0_rect * T.Tr_velo_to_cam * points_rotation')';
                px(:,1) = px(:,1)./px(:,3);
                px(:,2) = px(:,2)./px(:,3);
                img_xy = [min(px(:,1))+5, min(px(:,2)), max(px(:,1))+5, max(px(:,2))];
                if strcmp(label_data{1}{ii}, 'Cyclist')
                    bbox_img_cyclist = [bbox_img_cyclist; img_xy];
                else
                    if strcmp(label_data{1}{ii}, 'Pedestrian')
                        bbox_img_pedestrian = [bbox_img_pedestrian; img_xy];
                    else
                        bbox_img_car = [bbox_img_car; img_xy];
                        
                    end
                end
                
                
                
                
                
                
                
                z_index = 1;
                X_3D = [ point(1,z_index);  point(2,z_index);  point(3,z_index); point(4,z_index);  point(1,z_index); point(5,z_index);  point(6,z_index);  point(7,z_index);  point(8,z_index);  point(5,z_index);  point(6,z_index);  point(2,z_index);  point(3,z_index);  point(7,z_index);  point(8,z_index);  point(4,z_index)] ; 
                z_index = 2;
                Y_3D =  [ point(1,z_index);  point(2,z_index);  point(3,z_index); point(4,z_index);  point(1,z_index); point(5,z_index);  point(6,z_index);  point(7,z_index);  point(8,z_index);  point(5,z_index);  point(6,z_index);  point(2,z_index);  point(3,z_index);  point(7,z_index);  point(8,z_index);  point(4,z_index)] ;
                z_index = 3;
                Z_3D =  [ point(1,z_index);  point(2,z_index);  point(3,z_index); point(4,z_index);  point(1,z_index); point(5,z_index);  point(6,z_index);  point(7,z_index);  point(8,z_index);  point(5,z_index);  point(6,z_index);  point(2,z_index);  point(3,z_index);  point(7,z_index);  point(8,z_index);  point(4,z_index)] ;
                
                plot3(X_3D, Y_3D, Z_3D,'color','red');
 
                hold on;
            end
        end
        % project all points into the image plane
        
        % plot in the image
        delete(h1);
        h1 = figure(1);
        img_name = strrep(label_name, '.txt', '.png');
        [X2,map2] = imread(fullfile(img_folder_dir, img_name));
        imshow(X2)%subimage(X2);
        hold on 
        len_det = size(bbox_img_car, 1);
        for kk=1:len_det
            pos = [bbox_img_car(kk,1),bbox_img_car(kk,2),  bbox_img_car(kk,3) - bbox_img_car(kk,1), bbox_img_car(kk,4)-bbox_img_car(kk,2)];
           %pos = [100, 100, 50, 100]
           rectangle('Position',pos,'EdgeColor','r', 'LineWidth',3);
           hold on,
        end
        len_det = size(bbox_img_pedestrian, 1);
        for kk=1:len_det
            pos = [bbox_img_pedestrian(kk,1),bbox_img_pedestrian(kk,2),  bbox_img_pedestrian(kk,3) - bbox_img_pedestrian(kk,1), bbox_img_pedestrian(kk,4)-bbox_img_pedestrian(kk,2)];
           %pos = [100, 100, 50, 100]
           rectangle('Position',pos,'EdgeColor','b', 'LineWidth',3);
           hold on,
        end
        len_det = size(bbox_img_cyclist, 1);
        for kk=1:len_det
            pos = [bbox_img_cyclist(kk,1),bbox_img_cyclist(kk,2),  bbox_img_cyclist(kk,3) - bbox_img_cyclist(kk,1), bbox_img_cyclist(kk,4)-bbox_img_cyclist(kk,2)];
           %pos = [100, 100, 50, 100]
           rectangle('Position',pos,'EdgeColor','y', 'LineWidth',3);
           hold on,
        end
    end
end

function label_data = reading_label_data(label_folder_dir, label_name)
    
    fileID = fopen( fullfile(label_folder_dir, label_name),'r');
    %label_data = textscan(fileID,'%3s%5f%5f%5f%5f%6f%6f%f%[^\n\r]');
    label_data = textscan(fileID,'%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f %f','delimiter', ' ');
    fclose(fileID);
    
end


function point_cloud = prepare_point_cloud_data(point_cloud_folder_dir ,point_cloud_name)
    
    fid = fopen(fullfile(point_cloud_folder_dir, point_cloud_name),'rb');
    if fid < 1
        fprintf('No LIDAR files !!!\n');
        keyboard
    else
        point_cloud = fread(fid,[4 inf],'single')';
        fclose(fid);
    end
    
    index = and((point_cloud(:,2) < point_cloud(:,1) - 0.27), ( -point_cloud(:,2) < point_cloud(:,1) - 0.27));
    point_cloud = point_cloud(index,:);
    
end