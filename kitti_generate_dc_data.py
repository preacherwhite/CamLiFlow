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

        logging.info('Creating model: %s' % self.cfgs.model.name)
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    @torch.no_grad()
    def run(self):
        logging.info('Generating predictions...')
        self.model.eval()

        max_iterations = self.cfgs.testset.max_iterations
        
        # Initialize empty lists to store the 3D flows, point clouds, and RGB values
        flow_3d_outputs = []
        pc1_outputs = []
        pc1_rgb_outputs = []

        for i, inputs in enumerate(tqdm(self.test_loader)):
            if i >= max_iterations and max_iterations != -1:
                break

            inputs = copy_to_device(inputs, self.device)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.forward(inputs)

            for batch_id in range(size_of_batch(inputs)):
                flow_2d_pred = outputs['flow_2d'][batch_id]

                test_id = inputs['index'][batch_id].item()
                input_h = inputs['input_h'][batch_id].item()
                input_w = inputs['input_w'][batch_id].item()

                if self.save_outputs in ['all', 'flow']:
                    # Save predicted flow
                    flow_2d_pred = flow_2d_pred.permute(1, 2, 0).clamp(-500, 500).cpu().numpy()
                    flow_2d_pred = flow_2d_pred[:input_h, :input_w]
                    os.makedirs(f'{self.out_dir}/flow', exist_ok=True)
                    save_flow_png(f'{self.out_dir}/flow/%06d.png' % (test_id), flow_2d_pred)
                
                # Save 3d flow 
                flow_3d_pred = outputs['flow_3d'][batch_id].cpu().numpy()
                flow_3d_outputs.append(flow_3d_pred)

                # Save input point cloud for the first frame
                input_pcs = inputs['pcs'][batch_id].cpu().numpy().transpose()
                pc1 = input_pcs[:, :3]
                pc1_outputs.append(pc1)

                # Project point cloud onto image plane and find corresponding RGB values
                input_images = inputs['images'][batch_id].cpu().numpy()
                input_image1 = input_images[:3].transpose(1, 2, 0)

                f = inputs['intrinsics'][batch_id][0].item()
                cx = inputs['intrinsics'][batch_id][1].item()
                cy = inputs['intrinsics'][batch_id][2].item()

                pc1_rgb = np.zeros((pc1.shape[0], 3), dtype=np.uint8)

                for j in range(pc1.shape[0]):
                    x, y, z = pc1[j]
                    u = int(f * x / z + cx)
                    v = int(f * y / z + cy)

                    if 0 <= u < input_w and 0 <= v < input_h:
                        pc1_rgb[j] = input_image1[v, u]

                pc1_rgb_outputs.append(pc1_rgb)

                if self.save_outputs == 'all':
                    # Save input images
                    input_images = inputs['images'][batch_id].cpu().numpy()
                    input_image1 = input_images[:3].transpose(1, 2, 0)
                    input_image2 = input_images[3:].transpose(1, 2, 0)
                    os.makedirs(f'{self.out_dir}/image1', exist_ok=True)
                    os.makedirs(f'{self.out_dir}/image2', exist_ok=True)
                    cv2.imwrite(f'{self.out_dir}/image1/%06d.png' % (test_id), input_image1)
                    cv2.imwrite(f'{self.out_dir}/image2/%06d.png' % (test_id), input_image2)

                    # Save input depth maps
                    pc1 = input_pcs[:, :3]
                    pc2 = input_pcs[:, 3:]

                    input_depth1 = pc2depth(pc1, f, cx, cy)
                    input_depth2 = pc2depth(pc2, f, cx, cy)

                    os.makedirs(f'{self.out_dir}/depth1', exist_ok=True)
                    os.makedirs(f'{self.out_dir}/depth2', exist_ok=True)
                    cv2.imwrite(f'{self.out_dir}/depth1/%06d.png' % (test_id), input_depth1)
                    cv2.imwrite(f'{self.out_dir}/depth2/%06d.png' % (test_id), input_depth2)
        
        # Stack the 3D flows and point clouds into single arrays
        flow_3d_outputs = np.stack(flow_3d_outputs, axis=0)
        pc1_outputs = np.stack(pc1_outputs, axis=0)
        pc1_rgb_outputs = np.stack(pc1_rgb_outputs, axis=0)
        
        # Save the stacked 3D flows, point clouds, and RGB values
        os.makedirs(f'{self.out_dir}', exist_ok=True)
        np.save(f'{self.out_dir}/flow3d_outputs.npy', flow_3d_outputs)
        np.save(f'{self.out_dir}/pc1_outputs.npy', pc1_outputs)
        np.save(f'{self.out_dir}/pc1_rgb_outputs.npy', pc1_rgb_outputs)  

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