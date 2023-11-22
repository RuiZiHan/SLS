for DATASET in dmlab kitti dsprites_loc dsprites_ori smallnorb_azi
    do
        CUDA_VISIBLE_DEVICES=0 python sota_pruning.py --dataset $DATASET
    done
    
#all dataset
#cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele