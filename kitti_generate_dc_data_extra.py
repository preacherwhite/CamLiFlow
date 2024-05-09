import os
import utils
import hydra
import shutil
import logging
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig
from factory import model_factory
from models.utils import mesh_grid, knn_interpolation
from utils import copy_to_device, size_of_batch, save_flow_png,pc2depth
from kitti import KITTIDepthCompletion
import cv2
import numpy as np
from tqdm import tqdm
class Evaluator:
    def __init__(self, device: torch.device, cfgs: DictConfig, save_outputs, out_dir):
        self.cfgs = cfgs
        self.device = device
        self.save_outputs= save_outputs
        self.out_dir = out_dir
        logging.info('Loading test set from %s' % self.cfgs.testset.root_dir)
        self.test_dataset = KITTIDepthCompletion(self.cfgs.testset)

        self.test_loader = utils.FastDataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfgs.model.batch_size,
            num_workers=self.cfgs.testset.n_workers,
            pin_memory=True
        )

    @torch.no_grad()
    def run(self):
        logging.info('Savin extra data...')

        max_iterations = self.cfgs.testset.max_iterations
        for i, inputs in enumerate(tqdm(self.test_loader)):
            if i >= max_iterations and max_iterations != -1:
                break

            for batch_id in range(size_of_batch(inputs)):

                test_id = inputs['index'][batch_id].item()

                # Project point cloud onto image plane and find corresponding RGB values
                input_images = inputs['images'][batch_id].cpu().numpy()
                input_image1 = input_images[:3].transpose(1, 2, 0)

                f = inputs['intrinsics'][batch_id][0].item()
                cx = inputs['intrinsics'][batch_id][1].item()
                cy = inputs['intrinsics'][batch_id][2].item()

                # Save input images
                os.makedirs(f'{self.out_dir}/image', exist_ok=True)
                cv2.imwrite(f'{self.out_dir}/image/%06d.png' % (test_id), input_image1)
                # save intrinsics 
                os.makedirs(f'{self.out_dir}/intrinsics', exist_ok=True)
                np.save(f'{self.out_dir}/intrinsics/%06d.npy' % (test_id), inputs['intrinsics'][batch_id].cpu().numpy())

@hydra.main(config_path='conf', config_name='evaluator')
def main(cfgs: DictConfig):
    utils.init_logging()

    # change working directory
    shutil.rmtree(os.getcwd(), ignore_errors=True)
    os.chdir(hydra.utils.get_original_cwd())

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))
        cudnn.benchmark = True
    else:
        raise RuntimeError('Submission script does not support multi-GPU systems.')

    save_outputs = cfgs.get('save_outputs', 'flow')  # Default to 'all' if not specified
    out_dir = cfgs.get('out_dir', '/media/staging1/dhwang/kitti_dc_flow')  # Default to 'predictions' if not specified

    evaluator = Evaluator(device, cfgs, save_outputs, out_dir)
    evaluator.run()


if __name__ == '__main__':
    main()