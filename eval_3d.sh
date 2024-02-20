# python launch.py --export --gpu 0 --config configs/kplanes_sober_no_sds.yaml \
# resume=outputs/GSO_drunk_no_sds_hr/CHICKEN_RACER/ckpts/last.ckpt \
# system.exporter_type=mesh-exporter system.exporter.fmt=obj system.exporter.save_uv=false \
# system.geometry.isosurface_threshold=25. name=GSO_drunk_no_sds_hr tag=CHICKEN_RACER



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



# python eval_mesh.py --pr_dir=/weka/home-chunhanyao/one2345/GSO_drunk21/*.obj \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/ --rigid_align --vis_align \
# --output_file=/weka/home-chunhanyao/one2345/drunk_eval_3d.txt



exp_dir=GSO_drunk_gt
config_file=kplanes_drunk_gt.yaml
sober_or_drunk=drunk

# for dir in ../GSO_sober21/*/; do
#     python launch.py --export --gpu 0 --config configs/$config_file \
#     system.exporter_type=mesh-exporter system.exporter.fmt=obj system.geometry.isosurface_threshold=10. \
#     system.exporter.save_uv=false \
#     resume=outputs/$exp_dir/$(basename -- $dir)/ckpts/last.ckpt \
#     name=$exp_dir tag=$(basename -- $dir)
# done

python eval_mesh.py --pr_dir=outputs/$exp_dir/ \
--gt_mesh=/weka/adam/gso/*/model.obj \
--transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
--output_file=outputs/$exp_dir/eval_3d.txt



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
