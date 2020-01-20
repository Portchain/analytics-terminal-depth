import numpy as np
from portcall.data import AISTable
from portcall.portcall import BerthProbability
from portcall.terminal import Terminal, Quay
from skimage.draw import polygon2mask
from scipy import interpolate
from tqdm import tqdm as tqdm
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
from portcall.utils import spherical_distance_exact

from typing import List
from portcall.vessel import VesselTrack

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
            self._data = None
        else:
            self.set_data(data)

    def set_data(self, data: np.ndarray):
        assert data.ndim == 2
        assert self._first_points == data.shape[1]
        assert self._second_points == data.shape[0]
        self._data = data

    def get_data(self):
        return self._data

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
            image = np.logical_or(image, temp)
        return image

    def interpolate(self, lons: np.ndarray, lats: np.ndarray):
        data = self._data
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

    def contour(self, ax=None, fill=False, colorbar=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        data = self._data
        if fill:
            im = ax.contourf(self._first_axis, self._second_axis, data, cmap='ocean_r', **kwargs)
        else:
            im = ax.contour(self._first_axis, self._second_axis, data, cmap='ocean_r', **kwargs)

        if colorbar:
            fig = ax.get_figure()
            fig.colorbar(im, orientation=colorbar)

def rotate_vector(v, angle):
    angle = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

    return rot_mat @ v


def find_plateaus(y, min_value=5, half_width=1):
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
    def __init__(self, grid: MapGrid, n: int):
        self._base_grid = MapGrid(grid.get_first_axis(), grid.get_second_axis())
        dimensions = (n, *grid.map_size)
        assert len(dimensions) == 3
        self._data = np.tile(np.nan, dimensions).astype(np.float16)

    def insert_mask(self, layer, mask, value):
        draft = self._data[layer]
        draft[mask] = value
        self._data[layer] = draft

    def create_depth_map(self, count_limit=5, agg=None):
        if agg is None:
            agg = lambda x: np.nanquantile(x, 0.90, axis=0)

        depth_map = agg(self._data)

        counts = self.create_count_map().get_data()
        depth_map[counts < count_limit] = np.nan
        depth_map[np.isnan(depth_map)] = 0

        map_ = self._base_grid.copy()
        map_.set_data(depth_map)
        return map_

    def create_count_map(self):
        count_map = np.sum(~np.isnan(self._data), axis=0)

        map_ = self._base_grid.copy()
        map_.set_data(count_map)
        return map_


def create_draft_histogram_map(tracks:List[VesselTrack], terminal: Terminal, loa_buffer=30) -> (DraftContainer, DraftContainer):

    m = create_map_grid_of_terminal(terminal)

    # Extract historic drafts
    logger.info('Collect draft information in terminal')
    dynamic_drafts = DraftContainer(m, len(tracks))
    static_drafts = DraftContainer(m, len(tracks))

    for i, track in tqdm(enumerate(tracks), total=len(tracks)):
        polygons = [v.get_vessel_footprint(buffer=loa_buffer / 2) for v in track.get_vessel_generator()]
        dynamic_footprint = m.mask_of_polygons(polygons)

        draft_value = np.nanmax(track.draft)
        if not draft_value:
            draft_value = 1

        bp = BerthProbability(track, terminal)
        sub = track[bp.is_points_berthed() > 0.9]
        polygons = [v.get_vessel_footprint(buffer=loa_buffer / 2) for v in sub.get_vessel_generator()]
        static_footprint = m.mask_of_polygons(polygons)

        dynamic_drafts.insert_mask(i, dynamic_footprint, draft_value)
        static_drafts.insert_mask(i, static_footprint, draft_value)

    return static_drafts, dynamic_drafts


def import_ais_data(terminal, start_time, end_time):
    # Create a grid for the map
    logger.info('Initialize map grid')
    m = create_map_grid_of_terminal(terminal)
    logger.info('Load AIS data')
    with AISTable() as adb:
        tracks = adb.fetch_tracks(lat_lim=m.second_lim,
                                  lon_lim=m.first_lim,
                                  start_time=start_time,
                                  end_time=end_time)
        tracks = list(tracks)
    return tracks


def create_map_grid_of_terminal(terminal: Terminal):
    lon1, lat1 = terminal.outline.min(axis=0)
    lon2, lat2 = terminal.outline.max(axis=0)
    resolution = 10  # [m]
    map_width = spherical_distance_exact(np.array([lon1, lat1]), np.array([lon2, lat1]))
    map_height = spherical_distance_exact(np.array([lon1, lat1]), np.array([lon1, lat2]))
    n_lon = np.floor(map_width / resolution)
    n_lat = np.floor(map_height / resolution)
    logger.info('Calculating grid on map of size %d x %d with resolution of %.1f m' % (n_lon, n_lat, resolution))
    lon_axis = np.linspace(lon1, lon2, n_lon)
    lat_axis = np.linspace(lat1, lat2, n_lat)
    m = MapGrid(lon_axis, lat_axis)
    return m


def calculate_quay_aligned_map(quay: Quay, static_map: MapGrid): #TODO: change name to calculate_quay_aligned_map
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

    depth_2d = static_map.interpolate(grid_geodetic[..., 0], grid_geodetic[..., 1])

    # select the half of normal to quay where the water is
    positive_side = depth_2d[n_normal:]
    negative_side = depth_2d[:n_normal][::-1]

    if np.mean(positive_side) > np.mean(negative_side):
        depth_2d = positive_side
    else:
        depth_2d = negative_side

    m = MapGrid(axis_parallel, axis_normal, data=depth_2d)
    return m


def calculate_depth_profile(m: MapGrid):
    # extract 1D information
    depth_2d = m.get_data()
    depth_1d = np.nanquantile(depth_2d, 0.70, axis=0)
    axis_parallel = m.get_first_axis()
    return axis_parallel, depth_1d


def clean_depth_profile(position: np.ndarray, depth: np.ndarray):
    # discretize the depth
    depth = np.round(depth)

    # find plateaus
    select = find_plateaus(depth)
    clean_select = clean_bool_sequence(select, n=1)
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
