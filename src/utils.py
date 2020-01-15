import numpy as np
from portcall.data import AISTable
from portcall.portcall import BerthProbability
from portcall.terminal import Terminal, Quay
from skimage.draw import polygon2mask
from scipy import interpolate
from tqdm import tqdm as tqdm
import logging

logger = logging.getLogger(__name__)
from portcall.utils import spherical_distance_exact


class MapGrid:
    def __init__(self, first_axis, second_axis, data: (dict, None) = None):
        self._min_first = min(first_axis)
        self._max_first = max(first_axis)
        self._min_second = min(second_axis)
        self._max_second = max(second_axis)
        self._first_points = len(first_axis)
        self._second_points = len(second_axis)
        self._first_axis = first_axis
        self._second_axis = second_axis

        if data is None:
            self._data = dict()
        else:
            self._data = data

    def set_data_by_name(self, data: np.ndarray, name: str):
        assert data.ndim == 2
        assert self._first_points == data.shape[1]
        assert self._second_points == data.shape[0]
        self._data[name] = data

    def get_data_by_name(self, name: str):
        return self._data[name]

    def copy(self):
        return MapGrid(self._first_axis,
                       self._second_axis,
                       data=self._data)

    @property
    def map_size(self):
        return self._second_points, self._first_points

    @property
    def second_lim(self):
        return self._min_second, self._max_second

    @property
    def first_lim(self):
        return self._min_first, self._max_first

    def get_coordinate_grid(self):
        first_grid, second_grid = np.meshgrid(self._first_axis, self._first_axis)
        return first_grid, second_grid

    def get_indices_of_coordinates(self, points: np.ndarray):
        step_lon = (self._max_first - self._min_first) / self._first_points
        step_lat = (self._max_second - self._min_second) / self._second_points

        step_size = np.array([step_lon, step_lat])
        offset = np.array([self._min_first, self._min_second]) + step_size

        indices = np.round((points - offset) / step_size).astype('int')
        return indices[..., ::-1]

    def mask_of_polygons(self, polygons):
        image = np.zeros(self.map_size, dtype=bool)
        for polygon in polygons:
            idx_polygon = self.get_indices_of_coordinates(polygon)
            temp = polygon2mask(self.map_size, idx_polygon)  # TODO: speed up using opencv
            image = temp | image
        return image

    def interpolate(self, lons: np.ndarray, lats: np.ndarray, name: str):
        data = self._data[name]
        depth_func = interpolate.RectBivariateSpline(self._first_axis, self._second_axis, data.T)
        input_sz = lons.shape

        lons = lons.reshape((-1))
        lats = lats.reshape((-1))

        temp = depth_func.ev(lons, lats)

        return temp.reshape(input_sz)

    def get_first_axis(self):
        return self._first_axis

    def get_second_axis(self):
        return self._second_axis


def rotate_vector(v, angle):
    angle = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

    return rot_mat @ v


def find_plateaus(y, min_value=5):
    half_width = 1
    step = np.hstack((np.ones(half_width), -1 * np.ones(half_width)))
    peak = np.array([-0.5, 1, -0.5])
    d_step = np.convolve(y, step, mode='same')
    d_peak = np.convolve(y, peak, mode='same')

    upper_lim = 0.5
    lower_lim = -0.5

    select = (lower_lim <= d_step) & (d_step <= upper_lim) \
             & (lower_lim <= d_peak) & (d_peak <= upper_lim) \
             & (y > min_value)

    return select


def shrink_bool_sequence(seq, n=1):
    for _ in range(n):
        seq = (np.pad(seq, [1, 0], 'constant', constant_values=True)
               & np.pad(seq, [0, 1], 'constant', constant_values=True))[1:]
    return seq


def grow_bool_sequence(seq, n=1):
    for _ in range(n):
        seq = (np.pad(seq, [0, 1], 'constant', constant_values=False)
               | np.pad(seq, [1, 0], 'constant', constant_values=False))[:-1]
    return seq


def clean_bool_sequence(seq, n=3):
    temp = shrink_bool_sequence(seq, n=n)
    clean_seq = grow_bool_sequence(temp, n=n)
    return clean_seq


class DraftContainer:
    def __init__(self, dimensions: tuple):
        assert len(dimensions) == 3
        self._data = np.tile(np.nan, dimensions).astype(np.float16)

    def insert_mask(self, layer, mask, value):
        draft = self._data[layer]
        draft[mask] = value
        self._data[layer] = draft

    def create_depth_map(self):
        q = 0.90
        count_limit = 5
        depth_map = np.nanquantile(self._data, q, axis=0)
        # depth_dynamic = np.nanmax(dynamic_drafts,axis=0)
        count_map = np.sum(~np.isnan(self._data), axis=0)
        depth_map[count_map < count_limit] = np.nan
        depth_map[np.isnan(depth_map)] = 0
        return depth_map, count_map


def calculate_depth_map(terminal: Terminal,
                        start_time='2019-10-15',
                        end_time='2019-11-01'):
    # Create a grid for the map
    logger.info('Initialize map grid')
    lon1, lat1 = terminal.outline.min(axis=0)
    lon2, lat2 = terminal.outline.max(axis=0)

    resolution = 10  # [m]

    map_width = spherical_distance_exact(np.array([lon1, lat1]), np.array([lon2, lat1]))
    map_height = spherical_distance_exact(np.array([lon1, lat1]), np.array([lon1, lat2]))

    n_lon = np.floor(map_width/resolution)
    n_lat = np.floor(map_height/resolution)

    logger.info('Calculating grid on map of size %d x %d with resolution of %.1f m' % (n_lon, n_lat, resolution))
    lon_axis = np.linspace(lon1, lon2, n_lon)
    lat_axis = np.linspace(lat1, lat2, n_lat)
    m = MapGrid(lon_axis, lat_axis)

    logger.info('Load AIS data')
    with AISTable() as adb:
        tracks = adb.fetch_tracks(lat_lim=m.second_lim, lon_lim=m.first_lim,
                                  start_time=start_time,
                                  end_time=end_time)
        tracks = list(tracks)

    # Extract historic drafts
    logger.info('Collect draft information in terminal')
    dynamic_drafts = DraftContainer((len(tracks), *m.map_size))
    static_drafts = DraftContainer((len(tracks), *m.map_size))

    for i, track in tqdm(enumerate(tracks), total=len(tracks)):
        polygons = [v.get_terminal_footprint(buffer=30 / 2) for v in track.get_vessel_generator()]
        dynamic_footprint = m.mask_of_polygons(polygons)

        draft_value = np.nanmax(track.draft)
        if not draft_value:
            draft_value = 1

        bp = BerthProbability(track, terminal)
        track = track[bp.is_points_berthed()]
        polygons = [v.get_terminal_footprint(buffer=30 / 2) for v in track.get_vessel_generator()]
        static_footprint = m.mask_of_polygons(polygons)

        dynamic_drafts.insert_mask(i, dynamic_footprint, draft_value)
        static_drafts.insert_mask(i, static_footprint, draft_value)

    # Create depth and count maps
    logger.info('Create depth map')
    depth_static, count_static = static_drafts.create_depth_map()
    depth_dynamic, count_dynamic = dynamic_drafts.create_depth_map()

    static = m.copy()
    dynamic = m.copy()

    static.set_data_by_name(depth_static, name='depth')
    static.set_data_by_name(count_static, name='count')
    dynamic.set_data_by_name(depth_dynamic, name='depth')
    dynamic.set_data_by_name(count_dynamic, name='count')

    return static, dynamic


def calculate_depth_map_at_quay(quay: Quay, static_map: MapGrid):
    width = 100  # [m]
    resolution_parallel = 10  # [m]
    resolution_normal = 10  # [m]

    line_cart = quay.to_cartesian(quay.line)
    quay_vector = (line_cart[1] - line_cart[0])
    quay_length = np.linalg.norm(quay_vector)
    v_norm = quay_vector / quay_length
    o_norm = rotate_vector(v_norm, -90)
    origo = line_cart[0]

    axis_parallel = np.arange(0, quay_length, resolution_parallel)
    axis_normal = np.arange(0, width, resolution_normal)
    n_normal = len(axis_normal)
    axis_normal_double = np.hstack([-axis_normal[::-1], axis_normal])  # calc to both sides of quay since we not know
    # which side the water is on.

    normal_component = axis_normal_double[:, None] * o_norm[None]  # dim n_nor x 2
    parallel_component = axis_parallel[:, None] * v_norm[None]  # dim n_par x 2
    grid = normal_component[:, None] + parallel_component[None] + origo  # dim n_nor x n_par x 2

    grid_geodetic = quay.to_geodetic(grid)

    depth_2d = static_map.interpolate(grid_geodetic[..., 0], grid_geodetic[..., 1], 'depth')

    # select the half of normal to quay where the water is
    positive_side = depth_2d[n_normal:]
    negative_side = depth_2d[:n_normal][::-1]

    if np.mean(positive_side) > np.mean(negative_side):
        depth_2d = positive_side
    else:
        depth_2d = negative_side

    m = MapGrid(axis_parallel, axis_normal, {'depth': depth_2d})
    return m


def calculate_depth_profile(m: MapGrid):
    # extract 1D information
    depth_2d = m.get_data_by_name('depth')
    depth_1d = np.quantile(depth_2d, 0.90, axis=0)
    axis_parallel = m.get_first_axis()
    return axis_parallel, depth_1d


def clean_depth_profile(position: np.ndarray, depth: np.ndarray):
    # discretize the depth
    depth = np.round(depth)

    # find plateaus
    select = find_plateaus(depth)
    clean_select = clean_bool_sequence(select, n=3)
    clean_depth = depth[clean_select]
    clean_position = position[clean_select]

    # interpolate missing parts
    if len(clean_position) > 0:
        f = interpolate.interp1d(clean_position,
                                 clean_depth,
                                 kind='nearest',
                                 fill_value='extrapolate')

        filled_depth = f(position)
    else:
        filled_depth = np.tile(np.nan, depth.shape)

    return filled_depth
