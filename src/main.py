import argparse
from portcall.data import TerminalTable
from utils import create_draft_histogram_map, calculate_quay_aligned_map, clean_depth_profile, calculate_depth_profile
from utils import import_ais_data
from utils import convert_curve_to_sections
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=None)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info('Starting script')

    parser = argparse.ArgumentParser(description='Determine port calls and save to database')
    parser.add_argument("-s", "--start", dest='start_time',
                        default='2019-10-01', type=str,
                        help="Start time for AIS data used in analysis")
    parser.add_argument("-e", "--end", dest='end_time',
                        default='2019-11-01', type=str,
                        help="End time for AIS data used in analysis")
    parser.add_argument("-t", "--terminal", dest='terminal',
                        default='DEHAM-BUR', type=str,
                        help="Code of terminal to map quay depth")
    parser.add_argument("-v", "--verbose", dest='verbose', action='store_true', help="Enable verbosity")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    ais_start_time = args.start_time
    ais_end_time = args.end_time
    terminal = args.terminal
    verbose = args.verbose
    map_buffer = 0  # [m]
    resolution = 5  # [m]

    # Load terminal information
    logger.info('Load terminal information for %s' % terminal)
    with TerminalTable() as tdb:
        terminals = tdb.fetch_terminals(code=terminal)
        terminal = terminals[0]
    logger.info('Loaded terminal: %s' % str(terminal))

    tracks = import_ais_data(terminal, ais_start_time, ais_end_time)
    logger.info('Loaded AIS tracks of %d vessels' % len(tracks))
    static_drafts, dynamic_drafts = create_draft_histogram_map(tracks,
                                                               terminal,
                                                               resolution=resolution,
                                                               terminal_buffer=map_buffer)

    agg = lambda x: np.nanquantile(x, 0.95, axis=0)
    static_depth_map = static_drafts.create_depth_map(agg=agg)
    dynamic_depth_map = dynamic_drafts.create_depth_map(agg=agg)
    static_count_map = static_drafts.create_count_map()
    dynamic_count_map = dynamic_drafts.create_count_map()

    # Project depth info onto quay
    fig1, axs = plt.subplots(1, len(terminal.quays), figsize=(30, 10))
    if len(terminal.quays) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    depth_profiles = {}
    for i, quay in enumerate(terminal.quays):
        logger.info('Calculating depth for quay %s' % quay.code)
        quay_depth_map = calculate_quay_aligned_map(quay, static_depth_map)
        quay_position, quay_depth = calculate_depth_profile(quay_depth_map)
        quay_depth_cleaned = clean_depth_profile(quay_position, quay_depth, depth_unit=1)

        depth_profiles[quay.code] = convert_curve_to_sections(quay_position, quay_depth_cleaned, threshold=0.1)

        axs[i].plot(quay_position, quay_depth_cleaned, label='raw')
        axs[i].plot(quay_position, quay_depth, ':k', label='clean')

        axs[i].set_title('Quay: %s' % quay.code)
        axs[i].set_xlabel('Quay position [m]')
        axs[i].set_ylim(6, 20)
        axs[i].legend()

    fig2, axs = plt.subplots(2, 2, figsize=(30, 20))

    (ax11, ax12), (ax21, ax22) = axs
    terminal.plot(ax=ax22)
    static_depth_map.contour(ax=ax22, fill=True)
    terminal.plot(ax=ax12)
    dynamic_depth_map.contour(ax=ax12, fill=True)

    terminal.plot(ax=ax21)
    static_count_map.contour(ax=ax21, fill=True)
    terminal.plot(ax=ax11)
    dynamic_count_map.contour(ax=ax11, fill=True)

    ax11.set_title('Count', fontsize=18)
    ax12.set_title('Depth', fontsize=18)
    ax11.set_ylabel('Dynamic', fontsize=18)
    ax21.set_ylabel('Static', fontsize=18)

    # collect all depth profiles in single dataframe
    temp = []
    for quay_id, data in depth_profiles.items():
        temp.append(pd.DataFrame({'terminal': terminal.code,
                                  'quay': quay_id,
                                  'start_position': data[0],
                                  'depth': data[1],
                                  'length': data[2]}))
    df_profiles = pd.concat(temp, axis=0)

    save_data = True
    if save_data:
        filename = '{terminal}_quay_depth.png'.format(terminal=terminal.code)
        fig1.savefig(filename, dpi=200)
        filename = '{terminal}_map.png'.format(terminal=terminal.code)
        fig2.savefig(filename, dpi=200)
        filename = '{terminal}_depth.csv'.format(terminal=terminal.code)
        df_profiles.to_csv(filename, index=False)

    plt.show()

    # TODO: show quay depth on map
    # TODO: Indicate which end of quay we are counting from
    # TODO: add colorbar to plots
    # TODO: better aggregation function for drafts. max observed more than n times
    # TODO: I a track has multiple portcalls we are using the same draft for all of them. And that vessel is undercounted

    logger.info('Done!')
