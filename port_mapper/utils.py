from portcall.data import AISTable
import logging
import numpy as np

logger = logging.getLogger(__name__)


def import_ais_data(terminal, start_time, end_time):
    # Create a grid for the map
    lon1, lat1 = np.min(terminal.outline, axis=0)
    lon2, lat2 = np.max(terminal.outline, axis=0)
    logger.info('Load AIS data')
    with AISTable() as adb:
        tracks = adb.fetch_tracks(lat_lim=(lat1, lat2),
                                  lon_lim=(lon1, lon2),
                                  start_time=start_time,
                                  end_time=end_time)
        tracks = list(tracks)
    return tracks
