
from affectnet_validation import *
from gdl_apps.utils.load import load_model


def main():

    path_to_models = '/ps/finetune_deca'

    path_to_affectnet = "/home/rdanecek/Workspace/mount/project/EmotionalFacialAnimation/data/affectnet/"

    path_to_processed_affectnet = "/is/cluster/work/rdanecek/data/affectnet/"

    run_names = []

    run_names += ['2024_03_26_15-05-56'] # Detail with coarse

    mode = 'detail'
    # mode = 'coarse'

    for run_name in run_names:
        print(f"Beginning testing for '{run_name}' in mode '{mode}'")

        relative_to_path = None
        replace_root_path = None

        deca, conf = load_model(path_to_models, run_name, mode, relative_to_path, replace_root_path)

        run_name = conf[mode].inout.name

        deca.deca.config.resume_training = True
        deca.deca.config.pretrained_modelpath = "/home/admin123/Workspace//data/model.tar"
        deca.deca._load_old_checkpoint()
        run_name = "Original"

        deca.eval()

        import datetime
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        conf[mode].inout.random_id = str(hash(time))
        conf[mode].learning.logger_type = None
        conf['detail'].learning.logger_type = None
        conf['coarse'].learning.logger_type = None

        dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
        conf[mode].model.test_vis_frequency = 1
        conf[mode].inout.name = "afft_" + run_name
        single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net", dm=dm, project_name_="AffectNetTests")
        print("We're done y'all")


if __name__ == '__main__':
    main()
