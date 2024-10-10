
import sys
from gdl.datasets.AffectNetDataModule import AffectNetDataModule


def main(): 
    if len(sys.argv) < 2: 
        print("Usage: python process_affectnet.py <input_folder> <output_folder> <optional_processed_subfolder> <optional_subset_index>")

    downloaded_affectnet_folder = sys.argv[1]
    processed_output_folder = sys.argv[2]

    if len(sys.argv) >= 3: 
        processed_subfolder = sys.argv[3]
    else: 
        processed_subfolder = None


    if len(sys.argv) >= 4: 
        sid = int(sys.argv[4])
    else: 
        sid = None


    dm = AffectNetDataModule(
            downloaded_affectnet_folder,
            processed_output_folder,
            processed_subfolder=processed_subfolder,
            mode="manual",
            scale=1.25,
            ignore_invalid=True,
            )

    if sid is not None:
        if sid >= dm.num_subsets: 
            print(f"Subset index {sid} is larger than number of subsets.")
            sys.exit()
        dm._detect_landmarks_and_segment_subset(dm.subset_size * sid, min((sid + 1) * dm.subset_size, len(dm.df)))
    else:
        dm.prepare_data() 

    
    


if __name__ == "__main__":
    main()

