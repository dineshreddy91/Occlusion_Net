import torch


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            keypoints: (str): write your description
            size: (int): write your description
            mode: (todo): write your description
        """
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        """
        Crop the image.

        Args:
            self: (todo): write your description
            box: (array): write your description
        """
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        """
        Resize the size of the given size.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        """
        Transpose this object to another object.

        Args:
            self: (todo): write your description
            method: (str): write your description
        """
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")
        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        """
        Convert to a set of fields.

        Args:
            self: (todo): write your description
        """
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        """
        Get a fieldpoints for the given item

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        """
        Add a new field to the document.

        Args:
            self: (todo): write your description
            field: (todo): write your description
            field_data: (todo): write your description
        """
        self.extra_fields[field] = field_data

    def get_field(self, field):
        """
        Get the value of a field

        Args:
            self: (str): write your description
            field: (str): write your description
        """
        return self.extra_fields[field]

    def __repr__(self):
        """
        Return a repr representation of - recursively.

        Args:
            self: (todo): write your description
        """
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def _create_flip_indices(names, flip_map):
    """
    Create a list of indices of a set of a list.

    Args:
        names: (list): write your description
        flip_map: (dict): write your description
    """
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class VehicleKeypoints(Keypoints):
    NAMES = [
        'Right_Front_wheel',
	'Left_Front_wheel',
	'Right_Back_wheel',
	'Left_Back_wheel',
	'Right_Front_HeadLight',
	'Left_Front_HeadLight',
	'Right_Back_HeadLight',
	'Left_Back_HeadLight',
	'Exhaust',
	'Right_Front_Top',
	'Left_Front_Top',
	'Right_Back_Top',
	'Left_Back_Top',
        'Center'
    ]
    FLIP_MAP = {
        'Right_Front_wheel': 'Left_Front_wheel',
	'Right_Back_wheel': 'Left_Back_wheel',
	'Right_Front_HeadLight': 'Left_Front_HeadLight',
	'Right_Back_HeadLight': 'Left_Back_HeadLight',
	'Right_Front_Top': 'Left_Front_Top',
	'Right_Back_Top': 'Left_Back_Top'
    }


# TODO this doesn't look great
VehicleKeypoints.FLIP_INDS = _create_flip_indices(VehicleKeypoints.NAMES, VehicleKeypoints.FLIP_MAP)
def kp_connections(keypoints):
    """
    Return kp kp_connections

    Args:
        keypoints: (array): write your description
    """
    kp_lines = [[0, 2],
                [1, 3],
                [0, 1],
                [2, 3],
                [9, 11],
                [10, 12],
                [9, 10],
                [11, 12],
                [4, 0],
                [4, 9],
                [4, 5],
                [5, 1],
                [5, 10],
                [6, 2],
                [6, 11],
                [7, 3],
                [7, 12],
                [6, 7]
    ]
    return kp_lines
VehicleKeypoints.CONNECTIONS = kp_connections(VehicleKeypoints.NAMES)


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    """
    Convert a heatmap to a heatmap.

    Args:
        keypoints: (str): write your description
        rois: (todo): write your description
        heatmap_size: (int): write your description
    """
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)
def kp_connections_person(keypoints):
    """
    Return kp connection objects for a kp keypoints

    Args:
        keypoints: (str): write your description
    """
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections_person(PersonKeypoints.NAMES)
