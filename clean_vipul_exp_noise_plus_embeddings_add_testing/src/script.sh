python train.py  \
    --audio_dir ../data/audio_spilit_combine_wave_reduced/ \
    --image_batch 16 \
    --video_batch 16 \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator PatchVideoDiscriminator \
    --print_every 1 \
    --every_nth 2 \
    --batches 5000000 \
    --dim_z_content 50 \
    --dim_z_motion 10 \
    --ngpus 1 \
    --output_dir output \
    ../data/video_dataset/ ../data/video_dataset_test/ ../logs/urmp
