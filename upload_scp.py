import os
from joblib import Parallel, delayed
from tqdm import tqdm
import glob

if __name__ == '__main__':
    cases = ["20230203_12m"]
    ssh_name = '152.83.123.237'
    ident = 'li325'
    def upload_one(local_path, dest_path):
        os.system('scp -r {} {}@{}:{} > /dev/null'.format(local_path, ident, ssh_name, dest_path))


    for c in cases:
        local_path = str(c)
        # dest_path = '/datasets/work/d61-icvstaff/work/Experiments/Rodrigo/COVIU/sfm_benchmark/COLMAP/Iphone/PIS3/gantry/{}/aligned'.format(c)
        dest_path = "/home/li325/4T_data/xuesong_space/dataset/biomass/Cotton_MV_dataset/MV_imgs_colmap/{}".format(c)
        # file to move
        all_files = glob.glob('{}/*'.format(local_path))

        Parallel(n_jobs=6)(delayed(upload_one)(path, dest_path) for path in tqdm(all_files, desc='Uploading {}'.format(c), leave=False,total=len(all_files)))




