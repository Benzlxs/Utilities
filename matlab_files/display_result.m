%%
%reading
%plotting
%getinginput
%writting
clear;
clc;
%  root_dir = 'H:/Dataset/datasets/kitti/object/';
%  data_set = 'testing';
%  cam = 2; % 2 = left color camera
%  image_dir = fullfile(root_dir,[data_set '\image_' num2str(cam)],'\');

 %image_dir = '/home/ben/Downloads/presentation_data/image_02/';
 image_dir = '/home/ben/Dataset/conf_presentation_data/testing/image_2/';
 
 image  = dir([image_dir,'*.png']);
 
 detection_dir = ['/home/ben/Dataset/conf_presentation_data/step_311600'];%['/home/ben/prr_vis/2d_detection/'];
 %detection_dir = ['results_rrc/'];
 results = dir([detection_dir,'*.txt']);
 test_dir = './rr/';
 
 num_image = size(image,1);
 fig = figure(1)
  fig.CreateFcn = @movegui;
 start_ind =166 ;
 for ii = start_ind:1: num_image
     img = imread([image_dir, image(ii).name]);
        
     set(fig,'position',[0,0,  size(img,2), size(img,1)]);
     h1.axes = axes('position',[0,0,1, 1]);
     imshow(img,'parent',h1.axes)
     hold(h1.axes, 'on') 
     ii
     movegui(fig,'center');
     
     objects = readLabels(detection_dir, ii-1);
     gt_rois = []; 
     iter=1;
     for j = 1:size(objects,2)
           if objects(j).score > 0.05
             gt_rois( 1) = objects(j).x1;
             gt_rois(  2) = objects(j).y1;
             gt_rois( 3) = objects(j).x2;
             gt_rois( 4) = objects(j).y2;
             xxx = [gt_rois( 1); gt_rois( 3); gt_rois( 3); gt_rois( 1); gt_rois( 1)  ];
             yyy = [gt_rois( 2); gt_rois( 2); gt_rois(4); gt_rois( 4); gt_rois( 2)];
             plot( h1.axes, xxx , yyy , 'r' , 'LineWidth', 2); 
             text(h1.axes, gt_rois( 1), gt_rois( 2),num2str(objects(j).score),'Color','yellow');
             
             %waitforbuttonpress; 
             %key = get(gcf,'CurrentCharacter');
%             if  strcmp(lower(key), ' ')
%                      [x,y] = ginput(2);
%                      test_objects(iter).type  = 'Car';
%                      test_objects(iter).x1    = int32(x(1));
%                      test_objects(iter).y1    = int32(y(1));
%                      test_objects(iter).x2    = int32(x(2));
%                      test_objects(iter).y2    = int32(y(2));
%                      test_objects(iter).score = objects(j).score;
%                      iter = iter + 1;
%               end
           end
     end 
     name = [test_dir, strrep(image(ii).name,'.png','.jpg')]
     saveas(gcf,name)
%     writeLabels(test_objects,test_dir,ii-1);

     
 end
