# exp_name0=WildnFree_noBG_stable_Stable_Zero123
# exp_name=WildnFree_noBG_stable_Stable_Zero123_refine
# config0=configs/kplanes_no_sds_mark.yaml
# config=configs/kplanes_no_sds_refine_mark.yaml
# sober_or_drunk=sober
# data_root="/weka/proj-sv3d/EVAL_WildnFree_noBG_stable/Stable_Zero123/"
# # mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/Stable_Zero123"

# for dir in /weka/proj-sv3d/EVAL_WildnFree_noBG_stable/Stable_Zero123/*/; \
# do (
#     OBJ=$(basename -- $dir)
#     python launch.py --train --config=$config0 name=$exp_name0 tag=$OBJ \
#     data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
#     data.train_downsample_resolution=2 data.full_resolution_step=1000 \
#     data.use_omnidata_normals=False system.loss.lambda_normal=0.0 && \
#     # data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

#     python launch.py --train --config=$config \
#     system.geometry_convert_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
#     system.material_restore_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
#     name=$exp_name tag=$OBJ data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
#     trainer.max_steps=800 data.train_downsample_resolution=1 data.full_resolution_step=100 \
#     data.use_omnidata_normals=False system.loss.lambda_normal=0.0 && \
#     # data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

#     python launch.py --export --config=$config \
#     system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
#     system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
#     name=$exp_name tag=$OBJ data.dataroot=$data_root/$OBJ/
# ); done



# exp_name0=WildnFree_STUDY_Stable_Zero123
# exp_name=WildnFree_STUDY_Stable_Zero123_refine
# config0=configs/kplanes_no_sds_mark.yaml
# config=configs/kplanes_no_sds_refine_mark.yaml
# sober_or_drunk=sober
# data_root="/weka/proj-sv3d/EVAL_WildnFree_STUDY/Stable_Zero123/"
# # mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/Stable_Zero123"

# for dir in /weka/proj-sv3d/EVAL_WildnFree_STUDY/Stable_Zero123/*/; \
# do (
#     OBJ=$(basename -- $dir)
#     python launch.py --train --config=$config0 name=$exp_name0 tag=$OBJ \
#     data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
#     data.train_downsample_resolution=2 data.full_resolution_step=1000 \
#     data.use_omnidata_normals=False system.loss.lambda_normal=0.0 && \
#     # data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

#     python launch.py --train --config=$config \
#     system.geometry_convert_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
#     system.material_restore_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
#     name=$exp_name tag=$OBJ data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
#     trainer.max_steps=800 data.train_downsample_resolution=1 data.full_resolution_step=100 \
#     data.use_omnidata_normals=False system.loss.lambda_normal=0.0 && \
#     # data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

#     python launch.py --export --config=$config \
#     system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
#     system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
#     name=$exp_name tag=$OBJ data.dataroot=$data_root/$OBJ/
# ); done



exp_name0=OmniObject3D_Stable_Zero123
exp_name=OmniObject3D_Stable_Zero123_refine
config0=configs/kplanes_no_sds_mark.yaml
config=configs/kplanes_no_sds_refine_mark.yaml
sober_or_drunk=sober
data_root="/weka/proj-sv3d/EVAL_OmniObject3D/OmniObject3D_sober21/Stable_Zero123/"
# mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/Stable_Zero123"

for dir in /weka/proj-sv3d/EVAL_OmniObject3D/OmniObject3D_sober21/Stable_Zero123/o*/; \
do (
    OBJ=$(basename -- $dir)
    python launch.py --train --config=$config0 name=$exp_name0 tag=$OBJ \
    data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
    data.train_downsample_resolution=2 data.full_resolution_step=1000 \
    data.use_omnidata_normals=False system.loss.lambda_normal=0.0 && \
    # data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

    python launch.py --train --config=$config \
    system.geometry_convert_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
    system.material_restore_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
    name=$exp_name tag=$OBJ data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
    trainer.max_steps=800 data.train_downsample_resolution=1 data.full_resolution_step=100 \
    data.use_omnidata_normals=False system.loss.lambda_normal=0.0 && \
    # data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

    python launch.py --export --config=$config \
    system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=$exp_name tag=$OBJ data.dataroot=$data_root/$OBJ/
); done



# exp_name0=GSO_Stable_Zero123_static_static_no_sds
# exp_name=GSO_Stable_Zero123_static_static_no_sds_refine
# config0=configs/kplanes_no_sds_mark.yaml
# config=configs/kplanes_no_sds_refine_mark.yaml
# sober_or_drunk=sober
# data_root="/weka/proj-sv3d/EVAL_GSO300/GSO_sober21/Stable_Zero123/"
# mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/Stable_Zero123"

# for dir in ../GSO_sober21/*/; \
# do (
#     OBJ=$(basename -- $dir)
#     python launch.py --train --config=$config0 name=$exp_name0 tag=$OBJ \
#     data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
#     data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

#     python launch.py --train --config=$config \
#     system.geometry_convert_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
#     name=$exp_name tag=$OBJ data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
#     data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

#     python launch.py --export --config=$config \
#     system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
#     system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
#     name=$exp_name tag=$OBJ
# ); done

# python scripts/eval_mesh.py --pr_dir=outputs/$exp_name/ \
# --gt_mesh=/weka/adam/gso/*/model.obj \
# --transform_dir=/weka/proj-sv3d/DATASETS/GSO_${sober_or_drunk}21/ --rigid_align \
# --output_file=outputs/${exp_name}/eval_3d.txt

# python ../stable-research/scripts/threeD_diffusion/metrics_img2vid.py --name="sober" \
# --main_gt_dir=/weka/proj-sv3d/DATASETS/GSO_sober21/*/rgba \
# --main_pr_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}/*/save/sober \
# --out_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}

# python ../stable-research/scripts/threeD_diffusion/metrics_img2vid.py --name="drunk" \
# --main_gt_dir=/weka/proj-sv3d/DATASETS/GSO_drunk21/*/rgba \
# --main_pr_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}/*/save/drunk \
# --out_dir=/weka/home-chunhanyao/threestudio/outputs/${exp_name}
