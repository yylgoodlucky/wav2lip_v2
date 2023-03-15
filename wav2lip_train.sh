
CUDA_VISIBLE_DEVICES=5 python wav2lip_train.py \
--data_root /nfslocal/data/kaixinwang/LRS2_datasets/resolution_2x \
--checkpoint_dir /data/users/yongyuanli/work/wav2lip_improve_1/Wav2Lip/checkpoints_vggloss \
--syncnet_checkpoint_path /data/users/yongyuanli/work/Wav2Lip/expert_disc_checkpoint/lipsync_expert.pth \
# --checkpoint_path /data/users/yongyuanli/work/wav2lip_improve_1/Wav2Lip/checkpoints_se/checkpoint_step000250000.pth