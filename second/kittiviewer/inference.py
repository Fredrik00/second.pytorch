import os
import numpy as np
import time

import pickle
import sys
from functools import partial
from pathlib import Path

import second.core.box_np_ops as box_np_ops
import second.core.preprocess as prep
from second.core.box_coders import GroundBox3dCoder
from second.core.region_similarity import (DistanceSimilarity, NearestIouSimilarity, RotateIouSimilarity)
from second.core.sample_ops import DataBaseSamplerV2
from second.core.target_assigner import TargetAssigner
from second.data import kitti_common as kitti
from second.protos import pipeline_pb2
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.pytorch.inference import TorchInferenceContext
from second.utils.progress_bar import list_bar

class SecondBackend:
    def __init__(self):
        self.dt_annos = None
        self.inference_ctx = None

def build_network(BACKEND):
    BACKEND.inference_ctx = TorchInferenceContext()
    BACKEND.inference_ctx.build(BACKEND.config_path)
    BACKEND.inference_ctx.restore(BACKEND.checkpoint_path)
    print("build_network successful!")

def inference_by_input(BACKEND, points, calib, image_shape=None): # image shape as [h, w]
    rect = calib['R0_rect']
    P2 = calib['P2']
    Trv2c = calib['Tr_velo_to_cam']
    if image_shape is not None:
        points = box_np_ops.remove_outside_points(points, rect, Trv2c, P2, image_shape)
    wh = np.array(image_shape[::-1])
    whwh = np.tile(wh, 2)

    t = time.time()
    inputs = BACKEND.inference_ctx.get_inference_input_dict_v2(calib, image_shape, points)
    #print("inputs: ", inputs)
    print("input preparation time:", time.time() - t)
    t = time.time()
    with BACKEND.inference_ctx.ctx():
        dt_annos = BACKEND.inference_ctx.inference(inputs)[0]
    print("detection time:", time.time() - t)
    dims = dt_annos['dimensions']
    num_obj = dims.shape[0]
    loc = dt_annos['location']
    rots = dt_annos['rotation_y']
    labels = dt_annos['name']
    bbox = dt_annos['bbox'] / whwh

    dt_boxes_camera = np.concatenate(
        [loc, dims, rots[..., np.newaxis]], axis=1)
    dt_boxes = box_np_ops.box_camera_to_lidar(
        dt_boxes_camera, rect, Trv2c)
    box_np_ops.change_box3d_center_(dt_boxes, src=[0.5, 0.5, 0], dst=[0.5, 0.5, 0.5])
    locs = dt_boxes[:, :3]
    dims = dt_boxes[:, 3:6]
    rots = np.concatenate([np.zeros([num_obj, 2], dtype=np.float32), -dt_boxes[:, 6:7]], axis=1)

    annos = {"locs": locs.tolist(), "dims": dims.tolist(), "rots": rots.tolist(),
            "labels": labels.tolist(), "scores": dt_annos["score"].tolist(), "bbox": dt_annos["bbox"].tolist()}
    
    print("annos: ", annos)
    print("Inference complete")

    return annos

if __name__ == "__main__":
    BACKEND = SecondBackend()
 
    BACKEND.checkpoint_path = "/notebooks/second_models/all_test/voxelnet-74240.tckpt"
    BACKEND.config_path = "/notebooks/second_models/all_test/pipeline.config"

    image_shape = np.array([375, 1242])

    v_path = "/notebooks/DATA/Kitti/object/testing/velodyne/000001.bin"
    num_features = 4
    points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, num_features])
    
    calib = dict()
    calib['P2'] = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                            [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                            [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
                            [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]) 
    calib['R0_rect'] = np.array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
                                [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
                                [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])
    calib['Tr_velo_to_cam']= np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                                        [ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                                        [ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
                                        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])

    build_network(BACKEND)
    inference_by_input(BACKEND, points, calib, image_shape)