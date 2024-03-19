
exp_name0=WildnFree_noBG_stable_static
exp_name=WildnFree_noBG_stable_static_refine
config0=configs/kplanes_no_sds_mark.yaml
config=configs/kplanes_sds_refine_mark.yaml
sober_or_drunk=sober
data_root="/weka/home-chunhanyao/sv3d_eval/EVAL_WildnFree_noBG_stable_static/"
mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/EVAL_WildnFree_noBG_stable_static/"
svd_model="prediction_3D_OBJ_SVD21V_drunk"

# declare -a arr=("sdxl_00661_" "ComfyUI_temp_airdo_00512_" "ComfyUI_temp_nndxc_00002_" "ComfyUI_00093_" "renders-br-E25OJ4I7PoE-unsplash")
# declare -a arr=("ComfyUI_00093_")

# for OBJ in "${arr[@]}"; \
for dir in /weka/home-chunhanyao/sv3d_eval/EVAL_WildnFree_noBG_stable_dynamic/*/
do (
    OBJ=$(basename -- $dir)

    python launch.py --train --config=$config0 name=$exp_name0 tag=$OBJ \
    data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/rgb/ \
    data.omnidata_normal_path=$mono_normal_dir/$OBJ/mono_normal/ && \

    python launch.py --train --config=$config \
    system.geometry_convert_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
    system.material_restore_from=outputs/$exp_name0/$OBJ/ckpts/last.ckpt \
    name=$exp_name tag=$OBJ data.sober_or_drunk=$sober_or_drunk data.dataroot=$data_root/$OBJ/rgb/ \
    data.omnidata_normal_path=$mono_normal_dir/$OBJ/mono_normal/ \
    system.guidance.pretrained_model_name_or_path=$svd_model && \

    # python launch.py --test --config=configs/$config \
    # resume=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
    # name=$exp_name tag=$OBJ && \

    python launch.py --export --config=$config \
    system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=$exp_name tag=$OBJ data.dataroot=$data_root/$OBJ/rgb/
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
