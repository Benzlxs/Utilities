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
    % for testing data %
    %label_folder_dir = '/home/ben/Dataset/conf_presentation_data/step_311600';
    %point_cloud_folder_dir = '/home/ben/Dataset/conf_presentation_data/testing/velodyne_reduced';
    % for training data %
    label_folder_dir = '/home/ben/Dataset/KITTI/data_object_image_2/training/label_2';
    point_cloud_folder_dir = '/home/ben/Dataset/KITTI/velody_without_objects';
    
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
    test_dir = './rr/'
    for index = 9:num_data
        label_name = all_label(index).name ; 
        clearvars label_data
        label_data = reading_label_data(label_folder_dir, label_name);
            % plotting point cloud
            point_cloud_name = strrep(label_name, '.txt', '.bin');
            point_cloud = prepare_point_cloud_data(point_cloud_folder_dir ,point_cloud_name);
            idx = point_cloud(:,1) < 0; % x-axis
            point_cloud(idx,:) = [];
            idx = point_cloud(:,1)>70;
            point_cloud(idx,:) = [];
            idx = point_cloud(:,3)<-3;  % y - axis
            point_cloud(idx,:) = [];
            idx = point_cloud(:,3)> 1;
            point_cloud(idx,:) = [];
            idx = point_cloud(:,2)<-30;  % z -axis
            point_cloud(idx,:) = [];
            idx = point_cloud(:,2)> 30;
            point_cloud(idx,:) = [];                    
            delete(h1);
            h1 = figure(1);
            pcshow(point_cloud(:,1:3));
        
            %colormap summer
            colormap autumn%winter(10)
            set(gcf,'color','black');
            hold on;
            point = zeros(8,3);
            num_label = size(label_data{1,1},1)
           for ii = 1:num_label
            if  strcmp(label_data{1,1}(ii), 'Car')
%                 len  = label_data{1,2}(ii);
%                 hei = label_data{1, 4}(ii);
%                 wid  = label_data{1, 3}(ii);
%                 alph =  - label_data{1, 5}(ii);  %% clockwise direction
%                 ctr_x = label_data{1, 6}(ii);
%                 ctr_y = label_data{1 , 7}(ii);
%                 ctr_z = label_data{1 , 8}(ii);
                len  = label_data{1,11}(ii);
                hei = label_data{1, 10}(ii);
                wid  = label_data{1, 9}(ii);
                alph =  - label_data{1, 15}(ii);  %% clockwise direction
                
                CTR_PC = (inv(t_velo_cam)*[label_data{1, 12}(ii); label_data{1 , 13}(ii); label_data{1 , 14}(ii); 1])';
%                 ctr_x = label_data{1, 12}(ii);
%                 ctr_y = label_data{1 , 13}(ii);
%                 ctr_z = label_data{1 , 14}(ii);
                ctr_x = CTR_PC(1);
                ctr_y = CTR_PC(2);
                ctr_z = CTR_PC(3);

                retangle = [ wid/2,  len/2;
                                    wid/2, -len/2;
                                   -wid/2, -len/2;
                                   -wid/2,  len/2];  %(x, y)
                point(1:4,3) = ctr_z - hei/2;
                point(5:8,3) = ctr_z + hei/2;
                
                point(1,1:2) = [ X_after_rotation( wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( wid/2, len/2, alph) + ctr_y];
                point(2,1:2) = [ X_after_rotation( wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( wid/2, -len/2, alph) + ctr_y];
                point(3,1:2) = [ X_after_rotation( -wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( -wid/2, -len/2, alph) + ctr_y];
                point(4,1:2) = [ X_after_rotation( -wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( -wid/2, len/2, alph) + ctr_y];
                
                point(5,1:2) = [ X_after_rotation( wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( wid/2, len/2, alph) + ctr_y];
                point(6,1:2) = [ X_after_rotation( wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( wid/2, -len/2, alph) + ctr_y];
                point(7,1:2) = [ X_after_rotation( -wid/2, -len/2, alph) + ctr_x ,  Y_after_rotation( -wid/2, -len/2, alph) + ctr_y];
                point(8,1:2) = [ X_after_rotation( -wid/2, len/2, alph) +  ctr_x ,  Y_after_rotation( -wid/2, len/2, alph) + ctr_y];
                z_index = 1;
                X_3D = [ point(1,z_index);  point(2,z_index);  point(3,z_index); point(4,z_index);  point(1,z_index); point(5,z_index);  point(6,z_index);  point(7,z_index);  point(8,z_index);  point(5,z_index);  point(6,z_index);  point(2,z_index);  point(3,z_index);  point(7,z_index);  point(8,z_index);  point(4,z_index)] ; 
                z_index = 2;
                Y_3D =  [ point(1,z_index);  point(2,z_index);  point(3,z_index); point(4,z_index);  point(1,z_index); point(5,z_index);  point(6,z_index);  point(7,z_index);  point(8,z_index);  point(5,z_index);  point(6,z_index);  point(2,z_index);  point(3,z_index);  point(7,z_index);  point(8,z_index);  point(4,z_index)] ;
                z_index = 3;
                Z_3D =  [ point(1,z_index);  point(2,z_index);  point(3,z_index); point(4,z_index);  point(1,z_index); point(5,z_index);  point(6,z_index);  point(7,z_index);  point(8,z_index);  point(5,z_index);  point(6,z_index);  point(2,z_index);  point(3,z_index);  point(7,z_index);  point(8,z_index);  point(4,z_index)] ;
                plot3(X_3D, Y_3D, Z_3D,'color','blue','LineWidth',2);
 
                hold on;
            end
        end
         axis([0 70 -30 30 -5 5])
         %axis([0 50 -20 20 -2 1])
         set(gca,'XTick',[5 25 45 65])
         set(gca,'YTick',[-30 -10 10 30])
         set(gca,'ZTick',[-5 0 5])
         set(gcf, 'Position', get(0, 'Screensize'));
         grid off, 
         axis off
         zoom(3)
         view(250, 20)
         %set(gcf,'PaperUnits','inches','PaperPosition',[0 0 8 6],'color',[0.8, 0.8, 0.8])
         colormap Parula  
     name = [test_dir, strrep(label_name,'.txt','.jpg')]
     %saveas(gcf,name) 

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