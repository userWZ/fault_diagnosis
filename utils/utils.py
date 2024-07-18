import os
from models import model_identifier
def local_directory(name, model_cfg, diffusion_cfg, dataset_cfg, output_directory):
    # tensorboard_directory = train_cfg['tensorboard_directory']
    # ckpt_path = output_directory # train_cfg['output_directory']

    # generate experiment (local) path
    model_name = model_identifier(model_cfg)
    diffusion_name = f"_T{diffusion_cfg['T']}_betaT{diffusion_cfg['beta_T']}"
    
    if model_cfg["unconditional"]:
        data_name = ""
    else:
        data_name = f"_L{dataset_cfg['segment_length']}_hop{dataset_cfg['hop_length']}"
    local_path = model_name + diffusion_name + data_name + f"_{'uncond' if model_cfg['unconditional'] else 'cond'}"

    if not (name is None or name == ""):
        local_path = name + "_" + local_path

    # Get shared output_directory ready
    output_directory = os.path.join('exp', local_path, output_directory)
    os.makedirs(output_directory, exist_ok=True)
    print("output directory", output_directory, flush=True)
    return local_path, output_directory