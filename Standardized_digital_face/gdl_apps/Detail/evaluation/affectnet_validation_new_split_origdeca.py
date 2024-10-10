
from gdl_apps.training.test_and_finetune_deca import single_stage_deca_pass
from gdl_apps.utils.load import load_deca
from omegaconf import DictConfig, OmegaConf
import os, sys
from pathlib import Path
from gdl.datasets.AffectNetDataModule import AffectNetEmoNetSplitTestModule
from omegaconf import open_dict

def load_model(path_to_models,
              run_name,
              stage,
              relative_to_path=None,
              replace_root_path=None,
              mode='best',
              allow_stage_revert=False,
              ):
    run_path = Path(path_to_models) / run_name
    with open(Path(run_path) / "cfg.yaml", "r") as f:
        conf = OmegaConf.load(f)
    deca = load_deca(conf,
              stage,
              mode,
              relative_to_path,
              replace_root_path,
              terminate_on_failure= not allow_stage_revert
              )
    if deca is None and allow_stage_revert:
        deca = load_deca(conf,
                         "coarse",
                         mode,
                         relative_to_path,
                         replace_root_path,
                         )

    return deca, conf



def data_preparation_function(cfg,path_to_affect, path_to_processed_affect):
    dm = AffectNetEmoNetSplitTestModule(
            path_to_affect,
             path_to_processed_affect,
             # processed_subfolder="processed_2021_Apr_02_03-13-33",
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             mode="manual",
             scale=1.25,
             test_batch_size=1
    )
    return dm



def main():
    path_to_models = '/is/cluster/work/rdanecek/finetune_deca'
    path_to_affect = "/ps/project/EmotionalFacialAnimation/data/affect/"
    path_to_processed_affect = "/is/cluster/work/rdanecek/data/affect/"
    mode = 'detail'


    run_name = '2021_03_26_15-05-56_Orig'


    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    relative_to_path = None
    replace_root_path = None

    deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)

    deca.deca.config.resume_training = True
    deca.deca.config.pretrained_modelpath = "/lustre/home/rdanecek/workspace/repos/model.tar"
    deca.deca._load_old_checkpoint()
    run_name = "Original"

    dm = data_preparation_function(conf, path_to_affect, path_to_processed_affect)
    conf[mode].model.test_vis_frequency = 1

    with open_dict(conf["coarse"].model):
        conf["coarse"].model["deca_class"] = "EMO"
    with open_dict(conf["detail"].model):
        conf["detail"].model["deca_class"] = "EMO"
    conf[mode].inout.random_id = str(hash(time))
    conf[mode].inout.time = time
    conf[mode].inout.full_run_dir = str(Path( conf[mode].inout.output_dir) / (time + "_" + conf[mode].inout.random_id + "_" + conf[mode].inout.name) /  mode)
    conf[mode].inout.checkpoint_dir = str(Path(conf[mode].inout.full_run_dir) / "checkpoints")
    Path(conf[mode].inout.full_run_dir).mkdir(parents=True)

    print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_validation_new_split", dm=dm, project_name_="AffectTestsNewSplit")
    print("We're done y'all")

if __name__ == '__main__':
    main()
