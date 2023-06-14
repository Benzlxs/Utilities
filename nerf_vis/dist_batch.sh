#!/bin/bash
GREEN='\033[0;32;1m'
RED='\033[0;31;1m'
NOCOLOR='\033[0m'
force_clean=false
echo -e "${GREEN} INFO: start to process [$force_clean]"

num_cpu=25
gap=100

for img_idx in `seq 6 1 7`;
	do echo -e "${GREEN} processing image ${img_idx}"
	for i in `seq 0 ${gap} 1199`;
   		do echo -e "${GREEN}processing ${i}"
	    	python pc_batch_split.py ~/dataset/public_data/data/dtu_scan24 --column_start ${i} --column_gap ${gap} --img_idx ${img_idx} &
            sleep 10s
    		process_num=`ps -ef | grep pc_batch_split |wc -l`
	    	echo "process number is ${process_num}"
		    while [ ${process_num} -gt ${num_cpu} ]
    		do
	    		sleep 400s
		    	process_num=`ps -ef | grep pc_batch_split |wc -l`
    		done
	done
	# sleep 3h
done
