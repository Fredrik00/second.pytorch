from online_inference import SecondBackend, build_network, inference
import numpy as np
import csv

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BOX_COLOUR_SCHEME = {
    'Car': '#00FF00',           # Green
    'Pedestrian': '#00FFFF',    # Teal
    'Cyclist': '#FFFF00'        # Yellow
}

class ObjectLabel:
    """Object Label Class
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries

    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown

    1    alpha        Observation angle of object, ranging [-pi..pi]

    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location x,y,z in camera coordinates (in meters)

    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        self.type = ""  # Type of object
        self.truncation = 0.
        self.occlusion = 0.
        self.alpha = 0.
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True


def visualization(image, display=True, fig_size=(15, 9.15)):
    """Forms the plot figure and axis for the visualization

    Keyword arguments:
    :param image_dir -- directory of image files in the wavedata
    :param display -- display the image in non-blocking fashion
    :param fig_size -- (optional) size of the figure
    """

    def set_plot_limits(axes, image):
        # Set the plot limits to the size of the image, y is inverted
        axes.set_xlim(0, image.shape[1])
        axes.set_ylim(image.shape[0], 0)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

    # plot images
    ax1.imshow(image)
    ax2.imshow(image)

    set_plot_limits(ax1, image)
    set_plot_limits(ax2, image)

    if display:
        plt.show(block=False)

    return fig, ax1, ax2


def visualization_single_plot(image, display=True, fig_size=(12.9, 3.9)):
    """Forms the plot figure and axis for the visualization

    Keyword arguments:
    :param image_dir -- directory of image files in the wavedata
    :param img_idx -- index of the image file to present
    :param flipped -- flag to enable image flipping
    :param display -- display the image in non-blocking fashion
    :param fig_size -- (optional) size of the figure
    """

    # Create the figure
    fig, ax = plt.subplots(1, figsize=fig_size, facecolor='black')
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0,
                        hspace=0.0, wspace=0.0)

    # Set axes settings
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    # plot images
    ax.imshow(image)

    if display:
        plt.show(block=False)

    return fig, ax


def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d


def compute_box_corners_3d(object_label):
    """Computes the 3D bounding box corner positions from an ObjectLabel

    :param object_label: ObjectLabel to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # Compute rotational matrix
    rot = np.array([[+np.cos(object_label.ry), 0, +np.sin(object_label.ry)],
                    [0, 1, 0],
                    [-np.sin(object_label.ry), 0, +np.cos(object_label.ry)]])

    l = object_label.l
    w = object_label.w
    h = object_label.h

    # 3D BB corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + object_label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + object_label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + object_label.t[2]

    return corners_3d


def project_box3d_to_image(corners_3d, p):
    """Computes the 3D bounding box projected onto
    image space.

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix

    Returns:
        corners : numpy array of corner points projected
        onto image space.
        face_idx: numpy array of 3D bounding box face
    """
    # index for 3d bounding box face
    # it is converted to 4x4 matrix
    face_idx = np.array([0, 1, 5, 4,  # front face
                         1, 2, 6, 5,  # left face
                         2, 3, 7, 6,  # back face
                         3, 0, 4, 7]).reshape((4, 4))  # right face
    return project_to_image(corners_3d, p), face_idx


def compute_orientation_3d(obj, p):
    """Computes the orientation given object and camera matrix

    Keyword arguments:
    obj -- object file to draw bounding box
    p -- transform matrix
    """

    # compute rotational matrix
    rot = np.array([[+np.cos(obj.ry), 0, +np.sin(obj.ry)],
                    [0, 1, 0],
                    [-np.sin(obj.ry), 0, +np.cos(obj.ry)]])

    orientation3d = np.array([0.0, obj.l, 0.0, 0.0, 0.0, 0.0]).reshape(3, 2)
    orientation3d = np.dot(rot, orientation3d)

    orientation3d[0, :] = orientation3d[0, :] + obj.t[0]
    orientation3d[1, :] = orientation3d[1, :] + obj.t[1]
    orientation3d[2, :] = orientation3d[2, :] + obj.t[2]

    # only draw for boxes that are in front of the camera
    for idx in np.arange(orientation3d.shape[1]):
        if orientation3d[2, idx] < 0.1:
            return None

    return project_to_image(orientation3d, p)



def draw_box_2d(ax, obj, color_tm='g'):
    """Draws the 2D boxes given the subplot and the object properties

    Keyword arguments:
    :param ax -- subplot handle
    :param obj -- object file to draw bounding box
    """
    # draw the boxes
    # we also don't care about labels here
    rect = patches.Rectangle((obj.x1, obj.y1),
                                obj.x2 - obj.x1,
                                obj.y2 - obj.y1,
                                linewidth=2,
                                edgecolor=color_tm,
                                facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)


def draw_box_3d(ax, obj, p, show_orientation=True,
                color_table=None, line_width=3, double_line=True,
                box_color=None):
    """Draws the 3D boxes given the subplot, object label,
    and frame transformation matrix

    :param ax: subplot handle
    :param obj: object file to draw bounding box
    :param p:stereo frame transformation matrix

    :param show_orientation: optional, draw a line showing orientaion
    :param color_table: optional, a custom table for coloring the boxes,
        should have 4 values to match the 4 truncation values. This color
        scheme is used to display boxes colored based on difficulty.
    :param line_width: optional, custom line width to draw the box
    :param double_line: optional, overlays a thinner line inside the box lines
    :param box_color: optional, use a custom color for box (instead of
        the default color_table.
    """
    corners3d = compute_box_corners_3d(obj)
    corners, face_idx = project_box3d_to_image(corners3d, p)

    # define colors
    if color_table:
        if len(color_table) != 4:
            raise ValueError('Invalid color table length, must be 4')
    else:
        color_table = ["#00cc00", 'y', 'r', 'w']

    trun_style = ['solid', 'dashed']
    trc = int(obj.truncation > 0.1)

    if len(corners) > 0:
        for i in range(4):
            x = np.append(corners[0, face_idx[i, ]],
                          corners[0, face_idx[i, 0]])
            y = np.append(corners[1, face_idx[i, ]],
                          corners[1, face_idx[i, 0]])

            # Draw the boxes
            if box_color is None:
                box_color = color_table[int(obj.occlusion)]

            ax.plot(x, y, linewidth=line_width,
                    color=box_color,
                    linestyle=trun_style[trc])

            # Draw a thinner second line inside
            if double_line:
                ax.plot(x, y, linewidth=line_width / 3.0, color='b')

    if show_orientation:
        # Compute orientation 3D
        orientation = compute_orientation_3d(obj, p)

        if orientation is not None:
            x = np.append(orientation[0, ], orientation[0, ])
            y = np.append(orientation[1, ], orientation[1, ])

            # draw the boxes
            ax.plot(x, y, linewidth=4, color='w')
            ax.plot(x, y, linewidth=2, color='k')


def main(BACKEND, image, points, calib, idx):
    """This demo shows RPN proposals and AVOD predictions in 3D
    and 2D in image space. Given certain thresholds for proposals
    and predictions, it selects and draws the bounding boxes on
    the image sample. It goes through the entire proposal and
    prediction samples for the given dataset split.
    The proposals, overlaid, and prediction images can be toggled on or off
    separately in the options section.
    The prediction score and IoU with ground truth can be toggled on or off
    as well, shown as (score, IoU) above the detection.
    """
    annos = inference(BACKEND, points, calib, image.shape[:2], idx)

    pred_objects = [ObjectLabel() for prediction in annos["labels"]]

    # Fail if only one object?
    for i in range(len(pred_objects)):
        obj = pred_objects[i]

        obj.type = annos["labels"][i]
        obj.truncation = 0  # Not needed
        obj.occlusion = 0  # Not needed
        obj.alpha = annos["alpha"][i]  # Not needed
        obj.x1, obj.y1, obj.x2, obj.y2 = annos["bbox"][i]
        obj.w, obj.l, obj.h = annos["dims"][i]  # Different order from Kitti
        loc = annos["locs"][i]
        obj.t = (-loc[1], -loc[2] + obj.h/2, loc[0])  # Seems to be in lidar format initially with centroid not on ground plane
        obj.ry = -annos["rots"][i][2]  # Only value not 0, negative seems more correct

    fig_size = (10, 6.1)
    gt_classes = ['Car', 'Pedestrian', 'Cyclist']

    image_size = image.size

    #prop_fig, prop_2d_axes, prop_3d_axes = visualization(image, display=False)
    prop_fig, prop_3d_axes = visualization_single_plot(image, display=False)

    draw_predictions(pred_objects, None, prop_3d_axes, calib["P2"][:3])

    out_name = "/notebooks/second_output/images/%06d.png" % idx
    plt.savefig(out_name)
    plt.close(prop_fig)
    print('\nDone')


def draw_predictions(objects, prop_2d_axes, prop_3d_axes, p_matrix):
    # Draw filtered ground truth boxes
    for obj in objects:
        # Draw 2D boxes
        #draw_box_2d(prop_2d_axes, obj, color_tm='r')

        # Draw 3D boxes
        draw_box_3d(prop_3d_axes, obj, p_matrix,
                              show_orientation=False,
                              color_table=['r', 'y', 'r', 'w'],
                              line_width=2,
                              double_line=False)


def read_calibration(path):
    """Reads in Calibration file from Kitti Dataset.

    Keyword Arguments:
    ------------------
    calib_dir : Str
                Directory of the calibration files.

    img_idx : Int
              Index of the image.

    cam : Int
          Camera used from 0-3.

    Returns:
    --------
    frame_calibration_info : FrameCalibrationData
                             Contains a frame's full calibration data.

    """
    calib = dict()
    data_file = open(path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p = np.vstack((p, [0, 0, 0, 1]))
        p_all.append(p)

    calib["P2"] = p_all[2]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    tr_rect = np.reshape(tr_rect, (3, 3))
    tr_rect = np.insert(tr_rect, 3, 0, axis=1)
    calib["R0_rect"] = np.vstack((tr_rect, [0, 0, 0, 1]))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    tr_v2c = np.reshape(tr_v2c, (3, 4))
    calib['Tr_velo_to_cam'] = np.vstack((tr_v2c, [0, 0, 0, 1]))

    return calib


if __name__ == '__main__':
    BACKEND = SecondBackend()
    BACKEND.checkpoint_path = "/notebooks/second_models/carla_carped_finetune/voxelnet-62000.tckpt"
    BACKEND.config_path = "/notebooks/second_models/carla_carped_finetune/pipeline.config"
    build_network(BACKEND)

    for idx in range(2000, 2100):
        filename = "%06d" % idx
        dataset = "Arctic"

        # Loop below on more images
        image = np.array(Image.open("/notebooks/DATA/" + dataset + "/object/testing/image_2/" + filename + ".png"), dtype=np.uint8)

        v_path = "/notebooks/DATA/" + dataset + "/object/testing/velodyne/" + filename + ".bin"
        num_features = 4
        points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, num_features])
        
        calib = read_calibration("/notebooks/DATA/" + dataset + "/object/testing/calib/" + filename + ".txt")

        main(BACKEND, image, points, calib, idx)
