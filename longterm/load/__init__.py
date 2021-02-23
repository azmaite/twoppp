# Jonas Braun
# jonas.braun@epfl.ch
# 18.02.2021

import os
from pathlib import Path
from shutil import copyfile

import utils2p

NAS_DIR_LINUX = "/mnt/NAS"
LABSERVER_DIR = "/mnt/labserver"
LABSERVER_DIR_LH = os.path.join(LABSERVER_DIR, "HERMANS_Laura", "Experimental_data")
LABSERVER_DIR_LH_2P = os.path.join(LABSERVER_DIR_LH, "_2p")
LABSERVER_DIR_LH_BEH = os.path.join(LABSERVER_DIR_LH, "_behavior")


HOME_DIR = str(Path.home())
TMP_PROCESS_DIR = os.path.join(HOME_DIR, "tmp")
LOCAL_DATA_DIR = os.path.join(HOME_DIR, "data", "longterm")

TWOP_FOLDER = "2p"
PROCESSED_FOLDER = "processed"

RAW_GREEN_TIFF = "green.tif"
RAW_RED_TIFF = "red.tif"

def copy_remote_to_local(trial_dirs, target_base_dir=LOCAL_DATA_DIR, source_base_dir=LABSERVER_DIR_LH_2P, raw=True, xml=True, sync_dirs=None, beh_dirs=None):
    if not isinstance(trial_dirs, list):
        trial_dirs = [trial_dirs]
    if sync_dirs is not None:
        raise NotImplementedError
        if not isinstance(sync_dirs, list):
            sync_dirs = [sync_dirs]
        assert len(sync_dirs) == len(trial_dirs)
    if beh_dirs is not None:
        raise NotImplementedError
        if not isinstance(beh_dirs, list):
            beh_dirs = [beh_dirs]
        assert len(beh_dirs) == len(trial_dirs)

    for i_trial, source_trial_dir in enumerate(trial_dirs):
        source_abs_dir = os.path.join(source_base_dir, source_trial_dir)
        if not os.path.exists(source_abs_dir):
            raise FileNotFoundError
        target_trial_dir = os.path.join(target_base_dir, source_trial_dir)
        if not os.path.exists(target_trial_dir):
            os.makedirs(target_trial_dir)

        target_2p_dir = os.path.join(target_trial_dir, TWOP_FOLDER)
        if raw:
            if not os.path.exists(target_2p_dir):
                os.makedirs(target_2p_dir)
            raw_dir = utils2p.find_raw_file(source_abs_dir)
            _, file_name = os.path.split(raw_dir)
            copyfile(raw_dir, os.path.join(target_2p_dir, file_name))
        
        if xml:
            if not os.path.exists(target_2p_dir):
                os.makedirs(target_2p_dir)
            xml_dir = utils2p.find_metadata_file(source_abs_dir)
            _, file_name = os.path.split(xml_dir)
            copyfile(xml_dir, os.path.join(target_2p_dir, file_name))

def get_flies_from_datedir(date_dir, endswith="", contains=""):
    dir_list = os.listdir(date_dir)
    # return every subfolder that starts with 'Fly' and ends with user specified argument
    fly_dirs = [os.path.join(date_dir, folder) for folder in dir_list 
                    if not os.path.isfile(os.path.join(date_dir, folder)) 
                    and folder.startswith('Fly') 
                    and folder.endswith(endswith) 
                    and contains in folder
               ]
    return sorted(fly_dirs)

def get_trials_from_fly(fly_dir, startswith="", endswith="", contains="", exclude="processed"):
    if not isinstance(fly_dir, list):
        fly_dir = [fly_dir]

    dir_list = [os.listdir(this_dir) for this_dir in fly_dir]
    # return every subfolder that starts with "startswith" and ends with "endswith"
    trial_dirs = [[os.path.join(fly_dir[i_dir], folder) for folder in fly_dir_list 
                            if not os.path.isfile(os.path.join(fly_dir[i_dir], folder)) 
                            and folder.endswith(endswith) 
                            and folder.startswith(startswith)
                            and contains in folder
                            and not exclude in folder
                        ]
                       for i_dir, fly_dir_list in enumerate(dir_list)
                       ]
    return [sorted(this_dir) for this_dir in trial_dirs]

def load_trial(trial_dir):
    trial_xml = utils2p.find_metadata_file(trial_dir)
    trial_raw = utils2p.find_raw_file(trial_dir) 

    meta_data = utils2p.Metadata(trial_xml)
    green, red = utils2p.load_raw(path=trial_raw, metadata=meta_data)
    
    return (green,) if meta_data.get_gainB() == 0 else (green, red)

def convert_raw_to_tiff(trial_dir, overwrite=False, return_stacks=True):
    processed_dir = os.path.join(trial_dir, PROCESSED_FOLDER)
    if not os.path.exists(os.path.join(processed_dir)):
        os.makedirs(os.path.join(processed_dir))

    if os.path.isfile(os.path.join(processed_dir, RAW_GREEN_TIFF)) and not overwrite:
        if not return_stacks:
            return None, None
        green = utils2p.load_img(os.path.join(processed_dir, RAW_GREEN_TIFF))
        try:
            red = utils2p.load_img(os.path.join(processed_dir, RAW_RED_TIFF))
        except FileNotFoundError:
            red = None
            Warning("No red tif was found. Returning None. If you recorded it and want to create it, toggle the overwrite Flag")
        return green, red

    stacks = load_trial(trial_dir)
    if len(stacks) == 1:
        green = stacks[0]
        red = None
        utils2p.save_img(os.path.join(processed_dir, RAW_GREEN_TIFF), green)
    elif len(stacks) == 2:
        green, red = stacks
        utils2p.save_img(os.path.join(processed_dir, RAW_GREEN_TIFF), green)
        utils2p.save_img(os.path.join(processed_dir, RAW_RED_TIFF), red)
    else:
        raise NotImplementedError("More than two stacks are not implemented in load_experiment.")

    return green, red

if __name__ == "__main__":
    # trial_dirs = ["210212/Fly1/cs_003",
    #               "210212/Fly1/cs_005",
    #               "210212/Fly1/cs_007",
    #               "210212/Fly1/cs_010"]

    # copy_remote_to_local(trial_dirs=trial_dirs, source_base_dir=LABSERVER_DIR_LH_2P, target_base_dir=LOCAL_DATA_DIR, raw=True, xml=True)

    date_dir = os.path.join(LOCAL_DATA_DIR, "210212")
    fly_dirs = get_flies_from_datedir(date_dir)
    trial_dirs = get_trials_from_fly(fly_dirs)

    for fly_trial_dirs in trial_dirs:
        for trial_dir in fly_trial_dirs:
            convert_raw_to_tiff(trial_dir, return_stacks=False)



