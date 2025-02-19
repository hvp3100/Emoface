from gdl_apps.training.test_and_finetune_deca import single_stage_deca_pass
from gdl_apps.utils.load import load_model
import sys
from gdl.datasets.AffectNetDataModule import AffectNetTestModule


def data_preparation_function(cfg,path_to_emoface, path_to_processed_emofacenet):
    dm = AffectNetTestModule(
            path_to_emofacenet,
             path_to_processed_emofacenet,
             # processed_subfolder="processed_2021_Apr_02_03-13-33",
             processed_subfolder="processed_2024_Apr_05_15-22-18",
             mode="manual",
             scale=1.25,
             test_batch_size=1
    )
    return dm



def main():
    # path_to_models = '/ps/scratch/rdanecek/emoface/finetune_deca'
    path_to_models = '/is/cluster/work/rdanecek/emoface/finetune_deca'
    path_to_emonet = "/ps/project/EmotionalFacialAnimation/data/emonet/"
    path_to_processed_emonet = "/is/cluster/work/rdanecek/data/emonet/"
    run_name = sys.argv[1]

    if len(sys.argv) > 2:
        mode = sys.argv[2]
    else:
        mode = 'detail'
    deca, conf = load_model(path_to_models, run_name, mode, allow_stage_revert=True)

    deca.eval()

    dm = data_preparation_function(conf[mode], path_to_affectnet, path_to_processed_affectnet)
    conf[mode].model.test_vis_frequency = 1
    conf[mode].inout.name = "afft_" + conf[mode].inout.name
    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    conf[mode].inout.random_id = str(hash(time))
    print(f"Beginning testing for '{run_name}' in mode '{mode}'")
    single_stage_deca_pass(deca, conf[mode], stage="test", prefix="affect_net", dm=dm, project_name_="EmoNetTests")
    print("We're done y'all")


if __name__ == '__main__':
    main()
