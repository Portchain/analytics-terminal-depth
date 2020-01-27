import argparse
import os, sys
from portcall.data import TerminalTable
from port_mapper.depth_map import calculate_quay_aligned_map, create_draft_histogram_map
from port_mapper.depth_profile import calculate_depth_profile, clean_depth_profile, convert_curve_to_sections
from port_mapper.utils import import_ais_data
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
    parser.add_argument("-o", "--outdir", dest='output_dir',
                        default='outputs', type=str,
                        help="Output directory")

    parser.add_argument("-v", "--verbose", dest='verbose', action='store_true', help="Enable verbosity")
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    ais_start_time = args.start_time
    ais_end_time = args.end_time
    terminal = args.terminal
    output_dir = args.output_dir
    verbose = args.verbose

    map_type = 'static'

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
    drafts = create_draft_histogram_map(tracks,
                                        terminal,
                                        resolution=resolution,
                                        terminal_buffer=map_buffer,
                                        type=map_type)

    agg = lambda x: np.nanquantile(x, 0.95, axis=0)
    depth_map = drafts.create_depth_map(agg=agg)
    count_map = drafts.create_count_map()

    # Project depth info onto quay
    fig1, axs = plt.subplots(1, len(terminal.quays), figsize=(30, 10))
    if len(terminal.quays) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    depth_profiles = {}
    for i, quay in enumerate(terminal.quays):
        logger.info('Calculating depth for quay %s' % quay.code)
        quay_depth_map = calculate_quay_aligned_map(quay, depth_map)
        quay_position, quay_depth = calculate_depth_profile(quay_depth_map)
        quay_depth_cleaned = clean_depth_profile(quay_position, quay_depth, depth_unit=1)

        depth_profiles[quay.code] = convert_curve_to_sections(quay_position, quay_depth_cleaned, threshold=0.1)

        axs[i].plot(quay_position, quay_depth_cleaned, label='raw')
        axs[i].plot(quay_position, quay_depth, ':k', label='clean')

        axs[i].set_title('Quay: %s' % quay.code)
        axs[i].set_xlabel('Quay position [m]')
        axs[i].set_ylim(6, 20)
        axs[i].legend()

    fig2, axs = plt.subplots(1, 2, figsize=(30, 13))
    ax1, ax2 = axs
    terminal.plot(ax=ax1)
    count_map.contour(ax=ax1, fill=True)

    terminal.plot(ax=ax2)
    depth_map.contour(ax=ax2, fill=True)

    ax1.set_title('Count', fontsize=18)
    ax2.set_title('Depth', fontsize=18)

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = '{terminal}_quay_depth.png'.format(terminal=terminal.code)
        fig1.savefig(os.path.join(output_dir, filename), dpi=200)
        filename = '{terminal}_map.png'.format(terminal=terminal.code)
        fig2.savefig(os.path.join(output_dir, filename), dpi=200)
        filename = '{terminal}_depth.csv'.format(terminal=terminal.code)
        df_profiles.to_csv(os.path.join(output_dir, filename), index=False)

    plt.show()

    # TODO: show quay depth on map
    # TODO: Indicate which end of quay we are counting from
    # TODO: add colorbar to plots
    # TODO: better aggregation function for drafts. max observed more than n times
    # TODO: I a track has multiple portcalls we are using the same draft for all of them. And that vessel is undercounted

    logger.info('Done!')
