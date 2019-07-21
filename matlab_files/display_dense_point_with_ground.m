function display_all_points(base_dir)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Plots OXTS poses of a sequence
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)

% clear and close everything
clear all; close all; dbstop error; clc;
disp('======= KITTI DevKit Demo =======');

% sequence base directory
if nargin<1
  dense_base_dir = '/home/ben/Dataset/KITTI/dense_point_cloud';
  dense_base_dir= '/home/ben/Dataset/KITTI/dense_point_cloud';
  ground_plane_dir= '/home/ben/Dataset/KITTI/ground_plane/ground_plane_80_70_with_post_para';
  %ground_plane_dir= '/home/ben/Dataset/KITTI/ground_plane/ground_plane_80_70_without_post_para';
  
end
idx = 1242;%1242;% 1090; %

W = [-40, 40]; % [-30, 30] 
H = [0, 70];  % [0, 50]
w_sub = 4;   % 3
h_sub = 3.5;  % 2.5


figure(1), hold on
dense_point_cloud = read_point_cloud_2(dense_base_dir, idx) ;
rand_no = 80000;
sample = randi( size(dense_point_cloud,1) , 1 , rand_no);
in_pc = dense_point_cloud(sample,1:3);   % obtaining the coordinates of samples in point cloud
pcshow(in_pc(:,1:3));
view([270 90]); % Azimuth and Elevation angle
colormap colorcube

grid on
set(gca,'Xtick',0:3.5:70);
set(gca,'Ytick',-40:4:40);
set(gca,'GridAlpha',0.8);
set(gca,'LineWidth',1, 'GridLineStyle','-');
set(gcf, 'Position',  [50, 50, 2000, 1800]);

%ground_plane = read_ground_plane(ground_plane_dir, idx);
ground_plane = read_ground_plane_array(ground_plane_dir, idx);
w_size = ((W(2) - W(1))/w_sub);
h_size = (H(2) - H(1))/h_sub;
ground_plane = reshape(ground_plane, [h_size, w_size]);
ground_plane = ground_plane';
% plot the groud plane
num_plane =  size(ground_plane, 1);
for h_i = 1:h_size
    for w_j = 1:w_size
        z_z = ground_plane(h_i, w_j);
        if z_z < 9.5
            x_1 = H(1) + h_sub*(h_i-1);
            x_2 = x_1 + h_sub;
            y_1 = W(1) +  w_sub*(w_j-1);
            y_2 = y_1 + w_sub;
            X = [x_1, x_2, x_2, x_1];
            Y = [y_1, y_1, y_2, y_2];
            Z = [z_z, z_z, z_z, z_z];
            C = [rand, rand, rand];
            fill3(X,Y,Z,C);
            hold on
        end
    end
end

close all
offset = 1.0;
% removing the ground point
figure(2), hold on
for h_i = 1:h_size
    for w_j = 1:w_size
        z_z = ground_plane(h_i, w_j);
        if z_z < 9.5
            x_1 = H(1) + h_sub*(h_i-1);
            x_2 = x_1 + h_sub;
            y_1 = W(1) +  w_sub*(w_j-1);
            y_2 = y_1 + w_sub;
            index = (in_pc(:,1) < x_2)&(in_pc(:,1)>x_1)&(in_pc(:,2)<y_2)&(in_pc(:,2)>y_1)&(in_pc(:,3)>(z_z+offset));
            tt_pc = in_pc(index,:);
            pcshow(tt_pc(:,1:3));
            hold on
        end
    end
end
colormap colorcube






X = [10 20 20 10];
Y = [10 10 20 20];
Z = [-1 -1 -1 -1];
C = [rand, rand, rand];


end




function ground = read_ground_plane_array(base_dir, frame)
    fid = fopen(sprintf('%s/%06d.txt',base_dir,frame),'rb');
    %ground = fread(fid,[1 inf],'double')';
    ground = textscan(fid,'%f')';
    fclose(fid);
    ground = cell2mat(ground);
    %ground = cell2mat(ground);
    
end




function ground = read_ground_plane(base_dir, frame)
    fid = fopen(sprintf('%s/%06d.txt',base_dir,frame),'rb');
    ground = textscan(fid, '%f %f %f','Delimiter',',');
    fclose(fid);
    ground = cell2mat(ground);
    
end


function point_cloud = read_point_cloud_2(base_dir, frame)
    fid = fopen(sprintf('%s/%06d.bin',base_dir,frame),'rb');
    point_cloud = fread(fid,[4 inf],'single')';
    fclose(fid);
    %point_cloud = velo(1:5:end,:); % remove every 5th point for display speed
    %index = and((point_cloud(:,2) < point_cloud(:,1) - 0.27), ( -point_cloud(:,2) < point_cloud(:,1) - 0.27));
    %point_cloud = point_cloud(index,:);
    %point_cloud(:,4) = 1;
end

function point_cloud = read_point_cloud(base_dir, frame)
    fid = fopen(sprintf('%s/%06d.bin',base_dir,frame),'rb');
    point_cloud = fread(fid,[7 inf],'single')';
    fclose(fid);
    %point_cloud = velo(1:5:end,:); % remove every 5th point for display speed
    %index = and((point_cloud(:,2) < point_cloud(:,1) - 0.27), ( -point_cloud(:,2) < point_cloud(:,1) - 0.27));
    %point_cloud = point_cloud(index,:);
    %point_cloud(:,4) = 1;
end
