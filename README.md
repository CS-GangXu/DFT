# DFT
### Inference
1. create environment according to the 'environment.yaml'.
2. Please download the 'test_set.tar' from [Google Drive](https://drive.google.com/drive/folders/15PgDGO_E-ysyOnhdaW_abbeG-5y0biPX?usp=sharing).
3. Extract the image from the 'test_set.tar' and origanize the content as 'dataset/test_set/test_hdr' and 'dataset/test_set/test_sdr'.
4. Run inference.py to obtain the HDR images in 'output' folder.

### Training
1. create environment according to the 'environment.yaml'.
2. Please download the 'training_set.tar' from [Google Drive](https://drive.google.com/drive/folders/15PgDGO_E-ysyOnhdaW_abbeG-5y0biPX?usp=sharing).
3. Extract the image from the 'training_set.tar' and origanize the content as 'dataset/training_set/train_hdr_sub_c096_s240' and 'dataset/training_set/train_sdr_sub_c096_s240'.
4. Run 'python -opt options/train/config.yml'.
