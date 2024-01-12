import glob
import os
import sys

DATA_PATH = "/home/temp/multiview]"

MODELS_PATH = ["SVD21_FT4", "SVD21_FTcontinue", "SVD21FT_backUCG", "SVDMV_25frame21FT"]

THREESTUDIO_PATH = "/home/kplanes2/"

command = "--config /home/kplanes2/configs/kplanes_no_sds.yaml --train name=jenga tag=kplanes use_timestamp=False data.dataroot=/home/temp/multiview]/SVD21_FTcontinue/2_of_Jenga_Classic_Game_000.png"


job_paths = []

num_inputs = None

for model_path in MODELS_PATH:
    inputs = glob.glob(f"{DATA_PATH}/{model_path}/*")
    num_inputs = len(inputs)
    for input_path in inputs:
        job_paths.append(input_path)


job_paths_final = []
for i in range(num_inputs):
    job_paths_final += job_paths[i::num_inputs]


html_table = "<table>\n"

for i in range(len(job_paths_final) // 2):
    html_table += "  <tr>\n"
    for j in range(2):
        input_path = job_paths_final[i * 2 + j]
        input_name = os.path.basename(input_path)

        tag = os.path.basename(os.path.dirname(input_path))
        html_table += f'    <td><video><source src="kplanes2/outputs/{input_name}/{tag}/save/it2000.mp4"></video></td>\n'
    html_table += "  </tr>\n"

html_table += "</table>"

# Save the HTML to a file or print it
with open("index.html", "w") as file:
    file.write(html_table)


for input_path in job_paths_final:
    print(input_path)

    input_name = os.path.basename(input_path)

    tag = os.path.basename(os.path.dirname(input_path))

    os.system(
        f"cd {THREESTUDIO_PATH} && . venv/bin/activate && python launch.py --config  configs/kplanes_no_sds.yaml --train name={input_name} tag={tag} data.dataroot={input_path} use_timestamp=False"
    )
