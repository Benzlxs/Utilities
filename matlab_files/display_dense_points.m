function display_dense_points(base_dir)
% clear and close everything
clear all; close all; dbstop error; clc;
disp('======= Demo =======');

% sequence base directory
if nargin<1
  base_dir = '/home/ben/Dataset/KITTI/dense_point_cloud';
end

all_pc = dir(fullfile(base_dir,'*.bin'));
num_pc = length(all_pc)
figure(1),


for i=1:num_pc
    pc_dir = fullfile(all_pc(i).folder,all_pc(i).name);
    point_cloud = read_point_cloud(pc_dir);
    pcshow(point_cloud(:,1:3));
    view([270 90]); % Azimuth and Elevation angle
    colormap colorcube   
    pause(0.4);
end

end

function point_cloud = read_point_cloud(pc_dir)
    fid = fopen(pc_dir,'rb');
    point_cloud = fread(fid,[4 inf],'single')';
    fclose(fid);
end


