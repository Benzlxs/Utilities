%verfiication whether there is a order

function [ ]=main()
    clear, clc
    point_cloud_folder_dir = '/home/ben/Dataset/KITTI/data_object_image_2/training/velodyne';
    point_cloud_name = '000009.bin';
    point_cloud = prepare_point_cloud_data(point_cloud_folder_dir ,point_cloud_name);
    
    num_points = size(point_cloud,1);
    new_points_1 = zeros(num_points+1,1);
    new_points_1(1:num_points) = point_cloud(:,2);
    new_points_2 = zeros(num_points+1,1);
    new_points_2(2:(num_points+1)) = point_cloud(:,2);
    
    diff = new_points_2 - new_points_1;
   
    [rng, theta, alpha] = carteian2poloar_cc(point_cloud(:,1), point_cloud(:,2), point_cloud(:,3)); 
     
    plane_para = [0.00420528; 0.014928; 0.99987973; -1.05877695];
    dist2plane = point_cloud(:,1:3)*plane_para(1:3) - plane_para(4);
    idx=[];
    idx = find(dist2plane<0);
    %ground_point = point_cloud(idx,:);
    point_cloud(idx,:) = [];   
    [rng, theta, alpha] = carteian2poloar_cc(point_cloud(:,1), point_cloud(:,2), point_cloud(:,3)); 
 
end


function [range, eval_theta, rot_alpha] = carteian2poloar_cc(x, y, z)
    range = sqrt(x.^2 + y.^2 + z.^2);
    eval_theta = atan2(z, sqrt(x.^2 + y.^2));
    rot_alpha = acos(y./range);
    
    return 
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