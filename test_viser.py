
import os
import viser
import numpy as np
import torch
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import mathutils
import sys
import time

PI = 3.1415926

all_veggies = [
    "apple",
    "bell_pepper",
    "broccoli",
    "cabbage",
    "carrot",
    "garlic",
    "gourd",
    "mango",
    "orange",
    "pear",
    "potato",
    "pumpkin",
    "russet_potato",
    "sweet_pepper",
]

all_veggie_filepaths = [f"C:\\Users\\rohan\\Documents\\sp24\\cs280\\final_proj\\veggies\\pointclouds\\{veggie}.ply" for veggie in all_veggies]
all_texture_filepaths = [f"C:\\Users\\rohan\\Documents\\sp24\\cs280\\final_proj\\veggies\\textures\\{veggie}.jpg" for veggie in all_veggies]

scene_name = "roland_0"
scene_path = f"C:\\Users\\rohan\\Documents\\sp24\\cs280\\final_proj\\scenes\\{scene_name}.ply"

veggie_arrays = []
for i in range(len(all_veggies)):
    with open(all_veggie_filepaths[i], "rb") as f:
        veggie_cloud = PlyData.read(f)
    veggie_arrays.append(torch.tensor(np.stack((
                    np.asarray(veggie_cloud.elements[0]["x"]),
                    np.asarray(veggie_cloud.elements[0]["y"]),
                    np.asarray(veggie_cloud.elements[0]["z"])), axis=1)))


server = viser.ViserServer()
def import_object(scale, location, rotation, opacity, color, veggie_ind):
    veggie_scaled = veggie_arrays[int(veggie_ind)].clone() * scale
    server.add_point_cloud(
        f"/veggie_{location}",
        points=veggie_scaled.numpy(),
        colors=color,
        wxyz=rotation,
        position=location,
        point_size=0.001
    )


def parse_pointcloud(scene_path: str):
    with open(scene_path, "rb") as f:
        scene_cloud = PlyData.read(f)

    xyz = np.stack((np.asarray(scene_cloud.elements[0]["x"]),
                    np.asarray(scene_cloud.elements[0]["y"]),
                    np.asarray(scene_cloud.elements[0]["z"])), axis=1)
    xyz = torch.tensor(xyz)

    N = len(xyz)

    log_scales = np.stack((np.asarray(scene_cloud.elements[0]["scale_0"]),
                        np.asarray(scene_cloud.elements[0]["scale_1"]),
                        np.asarray(scene_cloud.elements[0]["scale_2"])), axis=1)

    scales = np.exp(log_scales)
    scales = torch.tensor(scales)

    quats = np.stack((np.asarray(scene_cloud.elements[0]["rot_0"]),
                        np.asarray(scene_cloud.elements[0]["rot_1"]),
                        np.asarray(scene_cloud.elements[0]["rot_2"]),
                        np.asarray(scene_cloud.elements[0]["rot_3"])), axis=1)
    quats = torch.tensor(quats)

    opacities = scene_cloud.elements[0]["opacity"]
    opacities = torch.tensor(opacities)

    veggie_inds = scene_cloud.elements[0]["veggie"]
    veggie_inds = torch.tensor(veggie_inds)

    colors = np.stack((np.asarray(scene_cloud.elements[0]["f_dc_0"]),
                        np.asarray(scene_cloud.elements[0]["f_dc_1"]),
                        np.asarray(scene_cloud.elements[0]["f_dc_2"])), axis=1)
    
    return xyz, scales, quats, opacities, colors, veggie_inds

locations, scales, rotations, opacities, colors, veggie_inds = parse_pointcloud(scene_path)

sparsity = 50 if len(sys.argv) < 2 else int(sys.argv[1])
opacity_bound = 0.5 if len(sys.argv) < 3 else float(sys.argv[2])

locations, scales, rotations, colors, veggie_inds, opacities = \
    locations[opacities > opacity_bound], \
    scales[opacities > opacity_bound], \
    rotations[opacities > opacity_bound], \
    colors[opacities > opacity_bound], \
    veggie_inds[opacities > opacity_bound], \
    opacities[opacities > opacity_bound]

suess_bbox = ((-0.3, 0), (0.4, 0.7), (-0.3, 0.2))
poster_bbox = ((-0.75, 0.1), (0, 1), (-0.8, 0))
roland_bbox = ((-0.5, 0.5), (-0.3, 0), (-1, 0.2))
bbox = roland_bbox
scales = scales[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
rotations = rotations[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
colors = colors[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
veggie_inds = veggie_inds[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
opacities = opacities[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
locations = locations[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]


scales, rotations, colors, veggie_inds, opacities, locations = \
    scales[::sparsity], \
    rotations[::sparsity], \
    colors[::sparsity], \
    veggie_inds[::sparsity], \
    opacities[::sparsity], \
    locations[::sparsity]

locations *= 5
scales *= 100


for i in tqdm(range(len(locations))):
    import_object(scales[i], locations[i], rotations[i], opacities[i], colors[i], veggie_inds[i])

while True:
    time.sleep(10.0)
