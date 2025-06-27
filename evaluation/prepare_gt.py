import json
from pathlib import Path
import pickle


# {classname: {meshname: [kp]}}

if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/IOS/3dteethland/data/3DTeethLand_landmarks_train')


    classes = ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp']
    out = {label: {} for label in classes}

    with open('fold_0.txt', 'r') as f:
        val_meshnames = [l.strip()[:-4] for l in f.readlines() if l.strip()]

    for kpt_file in root.glob('**/*.json'):
        meshname = kpt_file.name.split('__')[0]

        if meshname not in val_meshnames:
            continue

        with open(kpt_file, 'r') as f:
            kpt_dict = json.load(f)

        for label in classes:
            kpts = [kpt for kpt in kpt_dict['objects'] if kpt['class'] == label]
            out[label][meshname] = [kpt['coord'] for kpt in kpts]
            k = 3

    with open('gold_standard.pkl', 'wb') as f:
        pickle.dump(out, f)

