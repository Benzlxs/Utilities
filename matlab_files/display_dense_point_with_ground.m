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
  dense_base_dir= '/home/ben/Dataset/KITTI/bad_dense_point_cloud';
  ground_plane_dir= '/home/ben/Dataset/KITTI/ground_plane/ground_plane_with_postprocess';
  
end
idx = 61;

W = [-30, 30]; % [-40, 40] 
H = [0, 50];  % [0, 50]
w_sub = 3;
h_sub = 2.5;


figure(1), hold on
dense_point_cloud = read_point_cloud_2(dense_base_dir, idx) ;
pcshow(dense_point_cloud(:,1:3));
view([270 90]); % Azimuth and Elevation angle
colormap colorcube


%ground_plane = read_ground_plane(ground_plane_dir, idx);
ground_plane = read_ground_plane_array(ground_plane_dir, idx);
w_size = ((W(2) - W(1))/w_sub);
h_size = (H(2) - H(1))/h_sub;
ground_plane = reshape(ground_plane, [w_size, h_size])
% plot the groud plane
num_plane =  size(ground_plane, 1);
for h_i = h_size:-1:1
    for w_j = w_size:-1:1
        z_z = ground_plane(h_i, w_j);
        if z_z < 9.5
            x_1 = H(1) + h_sub*h_i;
            x_2 = x_1 + h_sub;
            y_1 = W(1) +  w_sub*w_j;
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
