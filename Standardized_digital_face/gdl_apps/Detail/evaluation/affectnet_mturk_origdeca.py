
import time as t
from affectnet_mturk import *
from omegaconf import open_dict



def main():

    path_to_models = '/is/cluster/work/rdanecek/finetune_deca'

    path_to_emoface = "/ps/project/EmotionalFacialAnimation/data/emoface/"

    path_to_processed_emoface = "/is/cluster/work/rdanecek/data/emoface/"

    run_names = []

    run_names += ['2021_03_26_15-05-56_Orig_DECA2']  # Detail with coarse

    mode = 'detail'


    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    mode = 'detail'

    relative_to_path = None
    replace_root_path = None

    for run_name in run_names:
        print(f"Beginning testing for '{run_name}' in mode '{mode}'")
        deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)


        deca.deca.config.resume_training = True

        deca.deca.config.pretrained_modelpath = "/lustre/home/rdanecek/workspace/repos/data/deca_model.tar"
        deca.deca._load_old_checkpoint()
        run_name = "Original"

        dm = data_preparation_function(conf, path_to_emoface, path_to_processed_emoface)
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

        single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net_mturk", dm=dm, project_name_="EmofaceMTurkTests",
                               )
        # t.sleep(3600)
        print("We're done y'all")


if __name__ == '__main__':
    main()
