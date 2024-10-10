import sys
import os 
from pathlib import Path
from typing import overload
import distutils.dir_util
from omegaconf import OmegaConf, DictConfig
import shutil
from gdl_apps.utils.load import load_model, replace_asset_dirs
from gdl.models.IO import locate_checkpoint

def package_model(input_dir, output_dir, asset_dir, overwrite=False, remove_bfm_textures=False):
    input_dir = Path(input_dir) 
    output_dir = Path(output_dir)
    asset_dir = Path(asset_dir)

    if output_dir.exists(): 
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Output directory '{output_dir}' already exists.")
            sys.exit()

    if not input_dir.is_dir(): 
        print(f"Input directory '{input_dir}' does not exist.")
        sys.exit()

    if not input_dir.is_dir(): 
        print(f"Input directory '{asset_dir}' does not exist.")
        sys.exit()

    with open(Path(input_dir) / "cfg.yaml", "r") as f:
        cfg = OmegaConf.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(input_dir / "cfg.yaml"), str(output_dir / "cfg.yaml"))
    checkpoints_dir = output_dir / "detail" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(locate_checkpoint(cfg["detail"], mode="best"))

    dst_checkpoint = checkpoints_dir / ( Path(checkpoint).relative_to(cfg.detail.inout.checkpoint_dir) )
    dst_checkpoint.parent.mkdir(parents=True, exist_ok=overwrite)
    shutil.copy(str(checkpoint), str(dst_checkpoint))


    cfg = replace_asset_dirs(cfg, output_dir)

    if remove_bfm_textures: 
        for mode in ["coarse", "detail"]:
            cfg[mode].model.use_texture = False


    with open(output_dir / "cfg.yaml", 'w') as outfile:
        OmegaConf.save(config=cfg, f=outfile)

    if remove_bfm_textures:
        emoface, _ = load_model(str(output_dir), output_dir, stage="detail")
        emoface._disable_texture(remove_from_model=True)
        from pytorch_lightning import Trainer
        trainer = Trainer(resume_from_checkpoint=dst_checkpoint)
        trainer.model = emoface
        trainer.save_checkpoint(dst_checkpoint)



def test_loading(outpath):
    outpath = Path(outpath)
    emoface = load_model(str(outpath.parent), outpath.name, stage="detail")
    print("Model loaded")

def main():

    if len(sys.argv) < 4:
        input_dir = "/ps/project/EmotionalFacialAnimation/emoface/face_reconstruction_models/new_affectnet_split/final_models" \
            "/2024_03-43-40_4753326650554236352_ExpDECA_Affec_clone_NoRing_EmoC_F2_DeSeggt_BlackC_Aug_early"
        output_dir = "/ps/project/EmotionalFacialAnimation/emoface/face_reconstruction_models/new_affectnet_split/final_models/packaged2/EMOFACE"
        asset_dir = "/home/rdanecek/Workspace/Repos/gdl/assets/"

    if len(sys.argv) >= 4:
        overwrite = bool(int(sys.argv[3]))
    else: 
        overwrite = True


    package_model(input_dir, output_dir, asset_dir, overwrite, remove_bfm_textures=True)
    print("Model packaged.")

    test_loading(output_dir)
    print("Model loading tested")


if __name__ == "__main__":
    main()