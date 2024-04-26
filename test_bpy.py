import bpy
import os
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import mathutils
import sys

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

scene_name = "water_bottle_0"
scene_path = f"C:\\Users\\rohan\\Documents\\sp24\\cs280\\final_proj\\scenes\\{scene_name}.ply"

# delete all current things in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

veggie_objects = []
for i in range(len(all_veggies)):
    bpy.ops.import_mesh.ply(filepath=all_veggie_filepaths[i])
    veggie_objects.append(bpy.context.selected_objects[0])

texture_objects = []
for i in range(len(all_veggies)):
    mat = bpy.data.materials.new(name=f"{all_veggies[i]}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(all_texture_filepaths[i])
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    texture_objects.append(mat)

def import_object(scale, location, rotation, opacity, veggie_ind):
    # Select the object
    veggie_ind = int(veggie_ind)
    bpy.ops.object.select_all(action='DESELECT')
    veggie_objects[veggie_ind].select_set(True)
    bpy.ops.object.duplicate()

    # Get reference to the imported object
    imported_object = bpy.context.selected_objects[0]

    # Set position and rotation
    imported_object.scale = scale
    imported_object.location = location
    imported_object.rotation_euler = rotation

    ob = imported_object

    # Assign it to object
    if ob.data.materials:
        ob.data.materials[0] = texture_objects[veggie_ind]
    else:
        ob.data.materials.append(texture_objects[veggie_ind])
    


def parse_pointcloud(scene_path: str):
    with open(scene_path, "rb") as f:
        scene_cloud = PlyData.read(f)

    xyz = np.stack((np.asarray(scene_cloud.elements[0]["x"]),
                    np.asarray(scene_cloud.elements[0]["y"]),
                    np.asarray(scene_cloud.elements[0]["z"])), axis=1)

    N = len(xyz)

    log_scales = np.stack((np.asarray(scene_cloud.elements[0]["scale_0"]),
                        np.asarray(scene_cloud.elements[0]["scale_1"]),
                        np.asarray(scene_cloud.elements[0]["scale_2"])), axis=1)

    scales = np.exp(log_scales)

    quats = np.stack((np.asarray(scene_cloud.elements[0]["rot_0"]),
                        np.asarray(scene_cloud.elements[0]["rot_1"]),
                        np.asarray(scene_cloud.elements[0]["rot_2"]),
                        np.asarray(scene_cloud.elements[0]["rot_3"])), axis=1)

    rots_euler = np.zeros((N, 3))

    for i in range(N):
        quat = mathutils.Quaternion(quats[i].tolist())
        euler = quat.to_euler()
        rots_euler[i] = (euler.x, euler.y, euler.z)
    
    opacities = scene_cloud.elements[0]["opacity"]

    veggie_inds = scene_cloud.elements[0]["veggie"]
    
    return xyz, scales, rots_euler, opacities, veggie_inds

locations, scales, rotations, opacities, veggie_inds = parse_pointcloud(scene_path)

sparsity = 50 if len(sys.argv) < 5 else int(sys.argv[4])
opacity_bound = 0.5 if len(sys.argv) < 6 else float(sys.argv[5])

locations, scales, rotations, veggie_inds, opacities = \
    locations[opacities > opacity_bound], \
    scales[opacities > opacity_bound], \
    rotations[opacities > opacity_bound], \
    veggie_inds[opacities > opacity_bound], \
    opacities[opacities > opacity_bound]

suess_bbox = ((-0.3, 0), (0.4, 0.7), (-0.3, 0.2))
poster_bbox = ((-0.75, 0.1), (0, 1), (-0.8, 0))
roland_bbox = ((-0.5, 0.5), (-0.3, 0), (-1, 0.2))
bear_bbox = ((-0.2, 0.2), (-1, 1), (-0.2, 0.2))
bear_1_bbox = ((-0.3, 0.3), (-1, 0), (-0.3, 0.3))

bbox = bear_1_bbox
# scales = scales[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
# rotations = rotations[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
# veggie_inds = veggie_inds[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
# opacities = opacities[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]
# locations = locations[(locations[:, 0] > bbox[0][0]) & (locations[:, 0] < bbox[0][1]) & (locations[:, 1] > bbox[1][0]) & (locations[:, 1] < bbox[1][1]) & (locations[:, 2] > bbox[2][0]) & (locations[:, 2] < bbox[2][1])]


scales, rotations, veggie_inds, opacities, locations = \
    scales[::sparsity], \
    rotations[::sparsity], \
    veggie_inds[::sparsity], \
    opacities[::sparsity], \
    locations[::sparsity]

locations *= 10
scales *= 200


for i in tqdm(range(len(locations))):
    import_object(scales[i], locations[i], rotations[i], opacities[i], veggie_inds[i])
    if i % 100 == 0 or i == len(locations) - 1:
        bpy.ops.object.select_all(action='SELECT')
        for obj in veggie_objects:
            obj.select_set(False)
        bpy.ops.object.join()

# delete og_pumpkin
bpy.ops.object.select_all(action='DESELECT')
for i in veggie_objects:
    i.select_set(True)
    bpy.ops.object.delete()

bpy.ops.wm.save_as_mainfile(filepath=f"C:\\Users\\rohan\\Documents\\sp24\\cs280\\final_proj\\{scene_name}.blend")

