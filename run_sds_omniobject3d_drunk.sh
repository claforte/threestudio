
exp_name0=OmniObject3D_drunk
exp_name=OmniObject3D_drunk_refine
config0=configs/kplanes_no_sds_mark.yaml
config=configs/kplanes_sds_refine_mark.yaml
sober_or_drunk=sober
data_root="/weka/proj-sv3d/EVAL_OmniObject3D/OmniObject3D_drunk21/SV3D_uncond_to_cond/"
mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/OmniObject3D_drunk21/"
svd_model="prediction_3D_OBJ_SVD21V_drunk"

for dir in /weka/proj-sv3d/EVAL_OmniObject3D/OmniObject3D_drunk21/SV3D_uncond_to_cond/c*/
do (
    OBJ=$(basename -- $dir)

    python launch.py --train --config=$config0 name=$exp_name0 tag=$OBJ \
    data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
    data.sober_orbit_path=/weka/proj-sv3d/DATASETS/OmniObject3D_sober21/$OBJ/ \
    data.drunk_orbit_path=/weka/proj-sv3d/DATASETS/OmniObject3D_drunk21/$OBJ/ \
    data.omnidata_normal_path=$mono_normal_dir/$OBJ/ && \

    python launch.py --train --config=$config \
    system.geometry_convert_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
    system.material_restore_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
    name=$exp_name tag=$OBJ data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/ \
    data.random_camera.cond_img_path=/weka/proj-sv3d/DATASETS/OmniObject3D_drunk21/$OBJ/ \
    data.omnidata_normal_path=$mono_normal_dir/$OBJ/ \
    system.guidance.pretrained_model_name_or_path=$svd_model && \

    python launch.py --export --config=$config \
    system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=$exp_name tag=$OBJ data.dataroot=$data_root/$OBJ/
); done

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
