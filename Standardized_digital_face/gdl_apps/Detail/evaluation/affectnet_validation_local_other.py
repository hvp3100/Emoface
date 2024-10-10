

from gdl.models.external.Deep3DFace import Deep3DFaceModule
from gdl.models.external.Face_3DDFA_v2 import Face3DDFAModule
import time as t
from affectnet_validation import *

def str2module(class_name):
    if class_name in ["3ddfa", "Face3DDFAModule"]:
        return Face3DDFAModule
    if class_name in ["deep3dface", "Deep3DFaceModule"]:
        return Deep3DFaceModule
    raise NotImplementedError(f"Not supported for {class_name}")


def instantiate_other_face_models(cfg, stage, prefix, checkpoint=None, checkpoint_kwargs=None):
    module_class = str2module(cfg.model.deca_class)

    if checkpoint is None:
        face_model = module_class(cfg.model, cfg.learning, cfg.inout, prefix)

    else:
        checkpoint_kwargs = checkpoint_kwargs or {}
        face_model = module_class.load_from_checkpoint(checkpoint_path=checkpoint, strict=False, **checkpoint_kwargs)
    return face_model



def main():
    path_to_models = '"/is/cluster/work/rdanecek/finetune'
    path_to_emonet = "/ps/project_cifs/data/emonet/"
    path_to_processed_emonet = "/is/cluster/work/rdanecek/data/emonet/"



    mode = 'detail'

    face_model = None
    from hydra.experimental import compose, initialize

    default = "deca_train_detail"
    overrides = [
        'model/settings=deep3dface',
        'learning/logging=none',
        'data/datasets=desktop',
        'data.num_workers=0',
        'learning.batch_size_train=4',
    ]

    initialize(config_path="../emoface_conf", job_name="test_face_model")
    conf = compose(config_name=default, overrides=overrides)

    print(f"Beginning testing for '{conf.model.deca_class}'.")

    import datetime
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

    conf.learning.logger_type = None

    dm = data_preparation_function(conf, path_to_affectnet, path_to_processed_affectnet)
    conf.model.test_vis_frequency = 1
    conf.inout.name = "afft_" + conf.model.deca_class
    conf.inout.random_id = str(hash(time))
    conf.inout.time = time
    conf.inout.full_run_dir = str(Path( conf.inout.output_dir) / (time + "_" + conf.inout.random_id + "_" + conf.inout.name) /  mode)
    conf.inout.checkpoint_dir = str(Path(conf.inout.full_run_dir) / "checkpoints")
    Path(conf.inout.full_run_dir).mkdir(parents=True)

    single_stage_deca_pass(face_model, conf, stage="test", prefix="emo_net", dm=dm, project_name_="EmoNetTests",
                           instantiation_function=instantiate_other_face_models)
    # t.sleep(3600)
    print("We're done y'all")


if __name__ == '__main__':
    main()
