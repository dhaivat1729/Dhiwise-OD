# Instruction to run inference on new images:

Requirement: Images you want to run inference on should be located in a particular directory. Path to where images should be stored should be provided too, the script will automatically create that new directory if it doesn't exist. 


```
cd ~/Dhiwise-OD
source ./DhiwiseOD/bin/activate
python3 tools/inference.py --options --inputImagesDir '/path/to/input/images' --modelPathDirectory 'experiments/datav3/R50_fasterRCNNv3_giou_loss/' --outputImageDir '/path/to/output/images'
```

## Example:

All DhiWise data images are located in `/home/ubuntu/data_OD/images`. If you want to run inference on that amd save it to `/home/ubuntu/data_OD/inferenceResults`, you can run the following command. 

```
python3 tools/inference.py --options --inputImagesDir '/home/ubuntu/data_OD/images' --modelPathDirectory 'experiments/datav3/R50_fasterRCNNv3_giou_loss/' --outputImageDir '/home/ubuntu/data_OD/inferenceResults'
```
