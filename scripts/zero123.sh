#debugpy-run launch.py -- \
python launch.py \
 --config=./configs/zero123_sai_multinoise_amb.yaml --train tag=Phase1_anya use_timestamp=false \
 system.loggers.wandb.enable=False system.loggers.wandb.project="lower_max_noise_0.5" system.loggers.wandb.name="lower_max_noise_0.5" \
 name=dummy-batch/anya_lower_max_noise_0.3_estimator_importance_res_512 data.default_elevation_deg=5 data.image_path=load/images/anya_front_rgba.png \
 system.freq.guidance_eval=8
