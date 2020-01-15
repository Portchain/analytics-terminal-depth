from portcall.data import TerminalTable
from utils import calculate_depth_map, calculate_depth_map_at_quay, clean_depth_profile, calculate_depth_profile
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=None)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info('Starting script')
    terminal = 'GBSOU-SCT'

    # Load terminal information
    logger.info('Load terminal information for %s' % terminal)
    with TerminalTable() as tdb:
        terminals = tdb.fetch_terminals(code=terminal)
        terminal = terminals[0]

    static_map, dynamic_map = calculate_depth_map(terminal,
                                                  start_time='2019-10-15',
                                                  end_time='2019-11-01')

    # Project depth info onto quay
    fig, axs = plt.subplots(1, len(terminal.quays))
    axs = axs.flatten()
    for i, quay in enumerate(terminal.quays):
        logger.info('Calculating depth for quay %s' % quay.code)
        quay_depth_map = calculate_depth_map_at_quay(quay, static_map)
        quay_position, quay_depth = calculate_depth_profile(quay_depth_map)
        quay_depth_cleaned = clean_depth_profile(quay_position, quay_depth)
        print(quay_depth_cleaned)
        axs[i].plot(quay_position, quay_depth_cleaned)
        axs[i].plot(quay_position, quay_depth, ':k')

    plt.show()

    logger.info('Done!')