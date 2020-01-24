import logging
import numpy as np
from scipy import interpolate

from port_mapper.depth_map import MapGrid

logger = logging.getLogger(__name__)


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


def calculate_depth_profile(m: MapGrid):
    # extract 1D information
    depth_2d = m.get_data()
    depth_1d = np.nanquantile(depth_2d, 0.95, axis=0)
    axis_parallel = m.get_first_axis()
    return axis_parallel, depth_1d


def clean_depth_profile(position: np.ndarray, depth: np.ndarray, depth_unit=1):
    # discretize the depth
    depth = np.round(depth / depth_unit) * depth_unit

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


def convert_curve_to_sections(x, y, threshold=0.1):
    section_start, section_value = [], []
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i == 0:
            section_start.append(xi)
            section_value.append(yi)
        else:
            if abs(yi - section_value[-1]) > threshold:
                section_start.append(xi)
                section_value.append(yi)

    section_start = np.array(section_start)
    section_value = np.array(section_value)
    section_length = np.diff(np.hstack([section_start, x[-1]]))

    return section_start, section_value, section_length