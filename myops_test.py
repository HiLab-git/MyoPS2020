# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys
from pymic.util.parse_config import parse_config
from pymic.net_run.agent_seg import SegmentationAgent
from path_config import path_dict

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('   python myops_test.py test config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    assert(stage == "test")
    if(not os.path.isfile(cfg_file)):
        print("configure file does not exist: {0:} ".format(cfg_file))
        exit()

    # reset data dir for configure
    config   = parse_config(cfg_file)
    data_dir = config['dataset']['root_dir']
    data_dir = data_dir.replace('MyoPS_data_dir', path_dict['MyoPS_data_dir']) 
    config['dataset']['root_dir'] = data_dir   
    agent = SegmentationAgent(config, stage)
    agent.run()

if __name__ == "__main__":
    main()