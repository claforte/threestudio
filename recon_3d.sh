
# for dir in ../GSO_sober21/*/; \
# do (python launch.py --export --gpu 0 --config configs/kplanes_sober_no_sds.yaml \
# resume=outputs/GSO_drunk_no_sds_hr/$(basename -- $dir)/ckpts/last.ckpt \
# system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
# system.exporter.save_uv=false name=GSO_drunk_no_sds_hr tag=$(basename -- $dir)); done


CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_sds_mark.yaml name=GSO_hashgrid_drunk_sds tag=CHICKEN_RACER \
    data.sober_or_drunk=drunk data.omnidata_normal_path=/weka/home-chunhanyao/GSO_mark/GSO_DRUNK_21/CHICKEN_RACER/mono_normal/

CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_sds_refine_mark.yaml \
    system.geometry_convert_from=outputs/GSO_hashgrid_drunk_sds/CHICKEN_RACER/ckpts/last.ckpt \
    name=GSO_hashgrid_drunk_sds_refine tag=CHICKEN_RACER data.sober_or_drunk=drunk \
    data.omnidata_normal_path=/weka/home-chunhanyao/GSO_mark/GSO_DRUNK_21/CHICKEN_RACER/mono_normal/

python launch.py --export --config=configs/kplanes_sds_refine_mark.yaml \
    system.geometry_convert_from=outputs/GSO_hashgrid_drunk_sds_refine/CHICKEN_RACER/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=GSO_hashgrid_drunk_sds_refine tag=CHICKEN_RACER



CUDA_VISIBLE_DEVICES=4,6 python launch.py --train --config=configs/kplanes_no_sds_mark.yaml name=GSO_hashgrid_drunk_no_sds tag=CHICKEN_RACER \
    data.sober_or_drunk=drunk data.omnidata_normal_path=/weka/home-markboss/GSO_DRUNK_21/CHICKEN_RACER/mono_normal/

CUDA_VISIBLE_DEVICES=4 python launch.py --train --config=configs/kplanes_no_sds_refine_mark.yaml \
    system.geometry_convert_from=outputs/GSO_hashgrid_drunk_no_sds/CHICKEN_RACER/ckpts/last.ckpt \
    name=GSO_hashgrid_drunk_no_sds_refine tag=CHICKEN_RACER data.sober_or_drunk=drunk \
    data.omnidata_normal_path=/weka/home-markboss/GSO_DRUNK_21/CHICKEN_RACER/mono_normal/


python launch.py --export --config=configs/kplanes_no_sds_mark.yaml \
    system.geometry_convert_from=outputs/GSO_hashgrid_drunk_no_sds/CHICKEN_RACER/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=GSO_hashgrid_drunk_no_sds tag=CHICKEN_RACER system.geometry.isosurface_threshold=50.

python launch.py --export --config=configs/kplanes_no_sds_refine_mark.yaml \
    system.geometry_convert_from=outputs/GSO_hashgrid_drunk_no_sds_refine/CHICKEN_RACER/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=GSO_hashgrid_drunk_no_sds_refine tag=CHICKEN_RACER

# ==========================
# ========= Recon ==========
# ==========================

### Train with sober GT
# CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_sober_gt.yaml name=GSO_sober_gt tag=CHICKEN_RACER
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_sober_gt.yaml name=GSO_sober_gt tag=$(basename -- $dir)); done

### Train with sober GT
# CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_sober_gt.yaml name=GSO_sober_gt tag=CHICKEN_RACER
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_drunk_gt.yaml name=GSO_drunk_gt tag=$(basename -- $dir)); done


### Eval for sober model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sober_gt/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sober_gt"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sober_gt/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sober_gt"

### Eval for drunk model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_gt/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_gt"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_gt/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_gt"


### Train with sober SVD
# CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_no_sds.yaml name=GSO_no_sds tag=CHICKEN_RACER
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_no_sds.yaml name=GSO_no_sds tag=$(basename -- $dir)); done

### Train with drunk SVD
# CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_drunk_no_sds.yaml name=GSO_drunk_no_sds tag=CHICKEN_RACER
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=6 python launch.py --train --config=configs/kplanes_drunk_no_sds.yaml name=GSO_drunk_no_sds_hr tag=$(basename -- $dir)); done

for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_drunk_cond_no_sds.yaml name=GSO_drunk_cond_no_sds tag=$(basename -- $dir)); done

cd /weka/home-chunhanyao/stable-research

### Eval for sober model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_no_sds/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_no_sds"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_no_sds/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_no_sds"

### Eval for drunk model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_no_sds/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_no_sds"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_no_sds/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_no_sds"

### Eval for drunk-cond model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_no_sds/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_no_sds"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_no_sds/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_no_sds"


# ==========================
# ====== Recon + SDS =======
# ==========================

### Train with sober SVD
# CUDA_VISIBLE_DEVICES=4,5 python launch.py --train --config=configs/kplanes_sds.yaml name=GSO_sds tag=CHICKEN_RACER
# CUDA_VISIBLE_DEVICES=4 python launch.py --test --config=configs/kplanes_sds.yaml name=GSO_sds tag=CHICKEN_RACER resume=outputs/GSO_sds/CHICKEN_RACER/ckpts/last.ckpt
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0,1 python launch.py --train --config=configs/kplanes_sds.yaml name=GSO_sds_new tag=$(basename -- $dir) && \
    CUDA_VISIBLE_DEVICES=0 python launch.py --test --config=configs/kplanes_sds.yaml name=GSO_sds_new tag=$(basename -- $dir) resume=outputs/GSO_sds_new/$(basename -- $dir)/ckpts/last.ckpt
); done

### Train with drunk SVD
# CUDA_VISIBLE_DEVICES=6,7 python launch.py --train --config=configs/kplanes_drunk_sds.yaml name=GSO_drunk_sds_test tag=CHICKEN_RACER
# CUDA_VISIBLE_DEVICES=0 python launch.py --test --config=configs/kplanes_drunk_sds.yaml name=GSO_drunk_sds tag=CHICKEN_RACER resume=outputs/GSO_drunk_sds/CHICKEN_RACER/ckpts/last.ckpt
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0,1 python launch.py --train --config=configs/kplanes_drunk_sds.yaml name=GSO_drunk_sds_hr tag=$(basename -- $dir) && \
    CUDA_VISIBLE_DEVICES=0 python launch.py --test --config=configs/kplanes_drunk_sds.yaml name=GSO_drunk_sds_hr tag=$(basename -- $dir) resume=outputs/GSO_drunk_sds_hr/$(basename -- $dir)/ckpts/last.ckpt
); done

for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=4,5 python launch.py --train --config=configs/kplanes_drunk_cond_sds.yaml name=GSO_drunk_cond_sds_hr tag=$(basename -- $dir) && \
    CUDA_VISIBLE_DEVICES=4 python launch.py --test --config=configs/kplanes_drunk_cond_sds.yaml name=GSO_drunk_cond_sds_hr tag=$(basename -- $dir) resume=outputs/GSO_drunk_cond_sds_hr/$(basename -- $dir)/ckpts/last.ckpt
); done


cd /weka/home-chunhanyao/stable-research

### Eval for sober model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sds_new/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sds_new"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sds_new/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_sds_new"

### Eval for drunk model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_sds_hr/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_sds_hr"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_sds_hr/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_sds_hr"

CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_sds_hr/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_sds_hr"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_sds_hr/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_sds_hr"


# ==========================
# ======== SDS only ========
# ==========================

### Train with sober SVD
# CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 python launch.py --train --config=configs/kplanes_only_sds.yaml name=GSO_only_sds tag=CHICKEN_RACER
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python launch.py --train --config=configs/kplanes_only_sds.yaml name=GSO_only_sds_new tag=$(basename -- $dir) && \
    CUDA_VISIBLE_DEVICES=0 python launch.py --test --config=configs/kplanes_only_sds.yaml name=GSO_only_sds_new tag=$(basename -- $dir) resume=outputs/GSO_only_sds_new/$(basename -- $dir)/ckpts/last.ckpt
); done

### Train with drunk SVD
# CUDA_VISIBLE_DEVICES=0 python launch.py --train --config=configs/kplanes_drunk_only_sds.yaml name=GSO_drunk_only_sds tag=CHICKEN_RACER
for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=4  python launch.py --train --config=configs/kplanes_drunk_only_sds.yaml name=GSO_drunk_only_sds_hr tag=$(basename -- $dir) && \
    CUDA_VISIBLE_DEVICES=1 python launch.py --test --config=configs/kplanes_drunk_only_sds.yaml name=GSO_drunk_only_sds_hr tag=$(basename -- $dir) resume=outputs/GSO_drunk_only_sds_hr/$(basename -- $dir)/ckpts/last.ckpt
); done


for dir in ../GSO_sober21/*/; \
do (CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4  python launch.py --train --config=configs/kplanes_drunk_cond_only_sds.yaml name=GSO_drunk_cond_only_sds_hr tag=$(basename -- $dir) && \
    CUDA_VISIBLE_DEVICES=6 python launch.py --test --config=configs/kplanes_drunk_cond_only_sds.yaml name=GSO_drunk_cond_only_sds_hr tag=$(basename -- $dir) resume=outputs/GSO_drunk_cond_only_sds_hr/$(basename -- $dir)/ckpts/last.ckpt
); done


### Eval for sober model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_only_sds_new/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_only_sds_new"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_only_sds_new/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_only_sds_new"

### Eval for drunk model
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_only_sds_new/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_only_sds_new"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_only_sds_new/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_only_sds_new"

CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="sober" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_only_sds_hr/*/save/sober" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_only_sds_hr"
CUDA_VISIBLE_DEVICES=0 python scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" --main_gt_dir="/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba" --main_pr_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_only_sds_hr/*/save/drunk" --out_dir="/weka/home-chunhanyao/threestudio/outputs/GSO_drunk_cond_only_sds_hr"
