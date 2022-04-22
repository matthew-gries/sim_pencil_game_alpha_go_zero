from typing import List
import os
import numpy as np
from PIL import Image
import pybullet as pb

def create_graph_image() -> np.ndarray:


    array = np.full(shape=(80, 80, 3), fill_value=255, dtype=np.uint8)

    array[12:17,38:42,:] = np.array([0, 0, 0])
    array[29:34,10:14,:] = np.array([0, 0, 0])
    array[46:51,10:14,:] = np.array([0, 0, 0])
    array[29:34,66:70,:] = np.array([0, 0, 0])
    array[46:51,66:70,:] = np.array([0, 0, 0])
    array[63:68,38:42,:] = np.array([0, 0, 0])
    return array

def add_board(
    grid_size: float=0.1, # CENTIMETERS!
) -> int:
    # create body
    coll_id = pb.createCollisionShape(
        pb.GEOM_BOX,
        halfExtents=(grid_size/2, grid_size/2, 0.01)
    )
    vis_id = pb.createVisualShape(pb.GEOM_BOX,
                                  halfExtents=(grid_size/2, grid_size/2, 0.01))
    obj_id = pb.createMultiBody(0, coll_id, vis_id)

    # TODO add collision object

    # place pattern in lower right corner of image, this seems to work
    # for adding textures to the top of any GEOM_BOX
    pattern = create_graph_image()
    padding = 3*pattern.shape[0]
    pattern = np.pad(pattern, ((padding,0),(padding,0),(0,0)))

    # save pattern as png so it can be imported by pybullet
    tex_fname = 'tmp_board_texture.png'
    Image.fromarray(pattern, mode="RGB").save(tex_fname)
    tex_id = pb.loadTexture(tex_fname)
    pb.changeVisualShape(obj_id, -1, textureUniqueId=tex_id,
                         rgbaColor=(1,1,1,1))

    os.remove(tex_fname)

    return obj_id