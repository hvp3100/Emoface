

from gdl_apps.EMOCA.training.test_and_finetune_deca import single_stage_deca_pass
from gdl_apps.EMOCA.utils.load import load_deca
from omegaconf import DictConfig, OmegaConf
import os, sys
from pathlib import Path
from gdl.datasets.AffectNetDataModule import AffectNetEmoNetSplitTestModule2


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



def data_preparation_function(cfg,path_to_affectnet, path_to_processed_affectnet):
    dm = AffectNetEmoNetSplitTestModule2(
            path_to_affectnet,
             path_to_processed_affectnet,
             processed_subfolder="processed_2021_Apr_05_15-22-18",
             mode="manual",
             scale=1.25,
             test_batch_size=1
    )
    return dm



def main():
    path_to_models = '/is/cluster/work/rdanecek/emoface/finetune_deca'
    path_to_affectnet = "/ps/project/EmotionalFacialAnimation/data/affectnet/"
    path_to_processed_affectnet = "/is/cluster/work/rdanecek/data/affectnet/"
    run_name = sys.argv[1]

    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'coarse'

    deca, conf = load_model(path_to_models, run_name, mode, allow_stage_revert=True)
    if mode != 'detail':
        mode = 'detail'
        deca.reconfigure(conf[mode].model, conf[mode].inout, conf[mode].learning, stage_name="",
                         downgrade_ok=False, train=False)
    deca.eval()

    import wandb
    api = wandb.Api()
    name = str(Path(run_name).name)
    idx = name.find("ExpDECA")
    run_id = name[:idx - 1]
    run = api.run("rdanecek/Emoface/" + run_id)
    tags = run.tags
    conf["coarse"]["learning"]["tags"] = tags
    conf["detail"]["learning"]["tags"] = tags
    tags += ["test"]

    dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
    conf[mode].model.test_vis_frequency = 1
    conf[mode].inout.name = conf[mode].inout.name
    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    conf[mode].inout.random_id = str(hash(time))
    print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net",
                           dm=dm, project_name_="AffectNetTestsNewSplit")
    print("We're done y'all")


if __name__ == '__main__':
    main()
