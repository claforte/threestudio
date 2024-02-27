# python launch.py --export --gpu 0 --config configs/kplanes_sober_no_sds.yaml \
# resume=outputs/GSO_drunk_no_sds_hr/CHICKEN_RACER/ckpts/last.ckpt \
# system.exporter_type=mesh-exporter system.exporter.fmt=obj system.exporter.save_uv=false \
# system.geometry.isosurface_threshold=25. name=GSO_drunk_no_sds_hr tag=CHICKEN_RACER


python eval_mesh.py --pr_dir=/weka/home-markboss/object_reconstructions/stable_zero123_drunk_gso50/*/Phase3/save/it0-export/model.obj \
--gt_mesh=/weka/adam/gso/*/model.obj \
--transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align --vis_align \
--output_file=outputs/stable_zero123_drunk_eval_3d.txt


python eval_mesh.py --pr_dir=/weka/home-chunhanyao/dreamgaussian/logs/GSO_drunk21_*_mesh.obj \
--gt_mesh=/weka/adam/gso/*/model.obj \
--transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align --vis_align \
--output_file=/weka/home-chunhanyao/dreamgaussian_drunk_eval_3d.txt


exp_name0=GSO_hashgrid_drunk_no_sds
exp_name=GSO_hashgrid_drunk_no_sds_refine
config0=configs/kplanes_no_sds_mark.yaml
config=configs/kplanes_no_sds_refine_mark.yaml
sober_or_drunk=drunk
mono_normal_dir="/weka/home-markboss/GSO_DRUNK_21"

for dir in ../GSO_sober21/*/; \
do (
    python launch.py --train --config=$config0 name=$exp_name0 tag=$(basename -- $dir) \
    data.sober_or_drunk=$sober_or_drunk data.omnidata_normal_path=$mono_normal_dir/$(basename -- $dir)/mono_normal/ && \

    python launch.py --train --config=$config \
    system.geometry_convert_from=outputs/$exp_name0/$(basename -- $dir)/ckpts/last.ckpt \
    name=$exp_name tag=$(basename -- $dir) data.sober_or_drunk=$sober_or_drunk \
    data.omnidata_normal_path=$mono_normal_dir/$(basename -- $dir)/mono_normal/ && \

    python launch.py --export --config=$config \
    system.geometry_convert_from=outputs/$exp_name/$(basename -- $dir)/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=$exp_name tag=$(basename -- $dir)
); done

python eval_mesh.py --pr_dir=outputs/$exp_name/ \
--gt_mesh=/weka/adam/gso/*/model.obj \
--transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
--output_file=outputs/${exp_name}/eval_3d.txt

python ../stable-research/scripts/threeD_diffusion/metrics_img2vid.py --name="sober" \
--main_gt_dir=/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba \
--main_pr_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}/*/save/sober \
--out_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}

python ../stable-research/scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" \
--main_gt_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba \
--main_pr_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}/*/save/drunk \
--out_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}




# python eval_mesh.py --pr_dir=/weka/home-chunhanyao/dreamgaussian/logs/GSO_sober21_*_mesh.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_sober21/ --rigid_align \
# --output_file=/weka/home-chunhanyao/dreamgaussian/logs/sober_eval_3d.txt

# python eval_mesh.py --pr_dir=/weka/home-chunhanyao/dreamgaussian/logs/GSO_drunk21_*_mesh.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align \
# --output_file=/weka/home-chunhanyao/dreamgaussian/logs/drunk_eval_3d.txt



# python eval_mesh.py --pr_dir=/weka/home-chunhanyao/shap-e/shap_e/outputs/GSO_sober21/*/model.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_sober21/ --rigid_align \
# --output_file=/weka/home-chunhanyao/shap-e/shap_e/outputs/sober_eval_3d.txt

# python eval_mesh.py --pr_dir=/weka/home-chunhanyao/shap-e/shap_e/outputs/GSO_drunk21/*/model.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align \
# --output_file=/weka/home-chunhanyao/shap-e/shap_e/outputs/drunk_eval_3d.txt



python eval_mesh.py --pr_dir=/weka/home-chunhanyao/point-e/point_e/outputs/GSO_drunk21/*/model.obj \
--gt_mesh=/weka/adam/gso/*/model.obj \
--transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align \
--output_file=/weka/home-chunhanyao/point-e/point_e/outputs/drunk_eval_3d.txt



# python eval_mesh.py --pr_dir=/weka/home-chunhanyao/one2345/GSO_drunk21/*.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align --vis_align \
# --output_file=/weka/home-chunhanyao/one2345/drunk_eval_3d.txt



# python eval_mesh.py --pr_dir=/weka/home-markboss/GSO_SOBER_21/*/Phase3/save/it0-export/model.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_sober21/ --rigid_align --vis_align \
# --output_file=outputs/$exp_dir/eval_3d.txt

# python eval_mesh.py --pr_dir=/weka/home-markboss/GSO_DRUNK_21/*/Phase3/save/it0-export/model.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align --vis_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_drunk_gt
# config_file=kplanes_drunk_gt.yaml
# sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_only_sds_new
# config_file=kplanes_sober_only_sds.yaml
# sober_or_drunk=sober

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_sds_new
# config_file=kplanes_sober_sds.yaml
# sober_or_drunk=sober

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_drunk_cond_no_sds
# config_file=kplanes_drunk_cond_no_sds.yaml
# sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt


# exp_dir=GSO_drunk_cond_only_sds_hr
# config_file=kplanes_drunk_cond_only_sds.yaml
# sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_drunk_cond_sds_hr
# config_file=kplanes_drunk_cond_sds.yaml
# sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_drunk_only_sds_hr
# config_file=kplanes_drunk_only_sds.yaml
# sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt



# exp_dir=GSO_drunk_sds_hr
# config_file=kplanes_drunk_sds.yaml
# sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

# python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/$exp_dir/eval_3d.txt
