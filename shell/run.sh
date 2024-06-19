# https://github.com/OpenDroneMap/docs/pull/133
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/cuda/11.3.0/targets/x86_64-linux/lib/stubs
#

# images_dir=/scratch3/li325/test2/code/images
# images_dir=/scratch3/li325/all_images/
images_dir=/scratch3/li325/first_flight/code/images
name=first_flight
output_dir=/scratch3/li325/$name
#mkdir -p $output_dir

singularity run \
--bind $images_dir:/$output_dir/code/images, \
--writable-tmpfs odm_latest.sif \
--orthophoto-png --mesh-octree-depth 12 --ignore-gsd --dtm \
--smrf-threshold 0.4 --smrf-window 24 --dsm --pc-csv --pc-las \
--orthophoto-kmz --ignore-gsd --matcher-type flann \
--max-concurrency 16 --use-hybrid-bundle-adjustment --build-overviews --time \
--dem-resolution 1.0 --orthophoto-resolution 0.1 --feature-quality ultra \
--pc-quality ultra --min-num-features 10000 --project-path $output_dir
