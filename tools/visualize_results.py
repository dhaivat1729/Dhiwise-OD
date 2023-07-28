"""
    This script will load predictions and groundtruths for a set of images and visualize in fiftyone.
"""

import fiftyone as fo
from DWODLib.utils import (load_json,
                        get_args,
                        isfile
                        )

from DWODLib.dataset import convert_json_from_list_to_dict, build_fiftyone_from_converted_json


## Usage: python3 tools/visualize_results.py --options --files ['file1.json', 'file2.json'] --filetypes ['pred', 'gt'] --imageDir "/path/to/image/directory"
## Example: python3 tools/visualize_results.py --options --files "['/home/ubuntu/yolooutput.json', '/home/ubuntuyolooutput.json']" --filetypes "['yolooutput', 'detectron2output']" --imageDir "/home/ubuntu/object-detection/data2/ToTest"

## get arguments
args = get_args()

## files
files = args['files']

## iterate through files and ensure it's a file
for file in files:
    assert isfile(file), "File %s does not exist." % file

## build converted jsons
imageKeyDicts = [convert_json_from_list_to_dict(file) for file in files]

## build fiftyone dataset
dataset = build_fiftyone_from_converted_json(imageKeyDicts, args['filetypes'], args['imageDir'])

## launch fiftyone app
session = fo.launch_app(dataset=dataset)
session.wait()




