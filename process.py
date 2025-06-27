import json
from pathlib import Path
# import tempfile
# tempfile.tempdir = '/output'

import pandas as pd

from infer import predict


if __name__ == '__main__':
    # predict(stage='landmarks', mixed=False, devices=1, config='teethland/config/config_synapse.yaml')

    # print('/input:', list(Path('/input').glob('*')))

    out_dir = Path('/output')
    out_dir = Path('preds/3dteethland_noweighted')
    out_dict = {k: [] for k in [
        'key', 'coord_x', 'coord_y', 'coord_z', 'class', 'score',
    ]}
    kpt_files = list(out_dir.glob('*__kpt.json'))
    for kpt_file in kpt_files:
        with open(kpt_file, 'r') as f:
            kpt_dict = json.load(f)
        for kpt in kpt_dict['objects']:
            out_dict['key'].append(kpt_dict['key'].split('/')[-1][:-4])
            out_dict['coord_x'].append(kpt['coord'][0])
            out_dict['coord_y'].append(kpt['coord'][1])
            out_dict['coord_z'].append(kpt['coord'][2])
            out_dict['class'].append(kpt['class'])
            out_dict['score'].append(kpt['score'])

    df = pd.DataFrame(out_dict)
    df.to_csv(out_dir / 'predictions.csv', index=False)

    print('Completed processing!')
