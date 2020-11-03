import itertools

from portcall.data import AISTable
import logging
import numpy as np

logger = logging.getLogger(__name__)


def import_ais_data(terminal, start_time, end_time, limit=None):
    lon1, lat1 = np.min(terminal.outline, axis=0)
    lon2, lat2 = np.max(terminal.outline, axis=0)
    logger.info('Load AIS data')
    with AISTable() as adb:
        tracks = adb.fetch_tracks(lat_lim=(lat1, lat2),
                                  lon_lim=(lon1, lon2),
                                  start_time=start_time,
                                  end_time=end_time)
        if limit is None:
            tracks = list(tracks)
        else:
            tracks = list(itertools.islice(tracks, limit))

    return tracks
