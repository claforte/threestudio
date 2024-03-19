
exp_name0=Otoy_dynamic
exp_name=Otoy_dynamic_refine
config0=configs/kplanes_no_sds_mark.yaml
config=configs/kplanes_sds_refine_mark.yaml
sober_or_drunk=sober
data_root="/weka/home-chunhanyao/sv3d_eval/Otoy_dynamic/"
mono_normal_dir="/weka/home-chunhanyao/sv3d_eval/Otoy_dynamic/"
svd_model="prediction_3D_OBJ_SVD21V_drunk"

# declare -a arr=("sdxl_00661_" "ComfyUI_temp_airdo_00512_" "ComfyUI_temp_nndxc_00002_" "ComfyUI_00093_" "renders-br-E25OJ4I7PoE-unsplash")
# declare -a arr=("Combatra")

# for dir in /weka/home-chunhanyao/sv3d_eval/Otoy/*/
# for OBJ in "${arr[@]}"; \

for dir in /weka/home-chunhanyao/sv3d_eval/Otoy_dynamic/*/; \
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

    python launch.py --export --config=$config \
    system.geometry_convert_from=outputs/$exp_name/$OBJ/ckpts/last.ckpt \
    system.exporter_type=mesh-exporter system.exporter.remesh_mesh=True system.exporter.clean_mesh=True \
    name=$exp_name tag=$OBJ data.dataroot=$data_root/$OBJ/rgb/
); done
