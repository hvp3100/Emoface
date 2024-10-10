
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parents[3] / "now_evaluation"))
from main import generating_cumulative_error_plots


def main():
    stage = 'coarse'


    run_names = {}
    run_names['Original'] = ["Paper EMOFACE", "detail"]


    run_names_new = {}
    run_names_new['2021_06_24_10-44-02_DECA__DeSegFalse_early'] = ["EMOFACE" , "detail"]

    run_names_new["2021_08_29_00-49-03_DECA_DecaD_EFswin_t_EDswin_t_DeSegrend_Deex_early"] = ["EMOFACE SWINT T", "detail"]
    run_names_new["2021_10_08_18-25-12_DECA_DecaD_NoRing_VGGl_DeSegrend_idBTH_Aug_early"]  = ["BTH", "coarse"]
    run_names_new["2021_10_08_16-40-04_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT-ft_Aug_early"]  = ["BT finetune id", "coarse"]
    run_names_new["2021_10_08_12-39-18_DECA_DecaD_NoRing_VGGl_DeSegrend_cosine_similarity_Aug_early"] = ["cos id", "coarse"]
    run_names_new["2021_10_13_10-49-29_DECA_DecaD_NoRing_VGGl_DeSegrend_idBT-ft_early"] =  ["BT finetune id v2 - on diag norm", "coarse"]
    run_names_new["2021_10_12_22-52-16_DECA_DecaD_VGGl_DeSegrend_idBT-ft-cont_Deex_early"] =  ["BT contrastive, ring 2, 0.2", "coarse"]
    run_names_new["2021_10_12_22-05-00_DECA_DecaD_VGGl_DeSegrend_idBT-ft-cont_Deex_early"] =  ["BT contrastive, ring 2, 0.3", "coarse"]
    run_names_new["2021_10_15_13-32-33_DECA__DeSegFalse_early"] =  ["EMOFACE, large batch", "coarse"]


    use_dense_topology = False

    path_to_old_models = '/home/rdanecek/Workspace/mount/scratch/rdanecek/emoface/finetune_deca'
    path_to_new_models = '/is/cluster/work/rdanecek/emoface/finetune_deca'


    run_files = []
    nicks = []
    path_to_models = path_to_old_models
    for run_name, nick in run_names.items():
        nick, stage = nick
        if use_dense_topology:
            savefolder = Path(path_to_models) / run_name / stage / "NoW_dense"
        else:
            savefolder = Path(path_to_models) / run_name / stage / "NoW_flame"

        run_files += [str(savefolder / "results" / "_computed_distances.npy")]
        nicks += [nick]

    path_to_models = path_to_new_models
    for run_name, nick in run_names_new.items():
        nick, stage = nick
        try:
            if use_dense_topology:
                savefolder = Path(path_to_models) / run_name / stage / "NoW_dense"
            else:
                savefolder = Path(path_to_models) / run_name / stage / "NoW_flame"
        except:
            if use_dense_topology:
                savefolder = Path(path_to_models) / run_name / 'coarse' / "NoW_dense"
            else:
                savefolder = Path(path_to_models) / run_name / 'coarse' / "NoW_flame"
        run_files += [str(savefolder / "results" / "_computed_distances.npy")]
        nicks += [nick]

    generating_cumulative_error_plots(run_files, nicks, "out.png")


if __name__ == "__main__":
    main()
