"""Run: python app/run_calibration_audit.py. Uses existing historical OOF only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from calibration_tools import isotonic, platt, metrics

root = Path(__file__).resolve().parents[1]
source = root/'outputs/ensemble/ensemble_oof.csv'
fold_source = root/'outputs/oof_predictions.csv'
d = pd.read_csv(source, dtype={'pipe_id': str})
folds = pd.read_csv(fold_source, dtype={'pipe_id': str})[['pipe_id','spatiotemporal_fold']]
assert not d.pipe_id.duplicated().any() and not folds.pipe_id.duplicated().any()
d = d.merge(folds, on='pipe_id', validate='one_to_one')
assert len(d) == 7288 and d.true_label.isin([0,1]).all()
assert d.ensemble_prob.between(0,1).all() and d.spatiotemporal_fold.notna().all()
assert d.spatiotemporal_fold.nunique() > 1
rows = [{'method': 'uncalibrated', **metrics(d.true_label, d.ensemble_prob)}]
fold_rows = []
for name, fn in [('platt', platt), ('isotonic', isotonic)]:
    pred = np.full(len(d), np.nan)
    for fold in sorted(d.spatiotemporal_fold.unique()):
        test = d.spatiotemporal_fold.eq(fold)
        train = ~test
        pred[test] = fn(d.loc[train,'ensemble_prob'], d.loc[train,'true_label'], d.loc[test,'ensemble_prob'])
        fold_rows.append({'method':name,'fold':int(fold),'train_n':int(train.sum()),'test_n':int(test.sum()),
                          **metrics(d.loc[test,'true_label'],pred[test])})
    assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
    d[name+'_crossfit_prob'] = pred
    rows.append({'method':name, **metrics(d.true_label,pred)})
out = root/'outputs/calibration_audit'
out.mkdir(parents=True,exist_ok=True)
d.to_csv(out/'crossfit_predictions.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False)
manifest = {'source': str(source.relative_to(root)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'fold_source': str(fold_source.relative_to(root)), 'fold_sha256':hashlib.sha256(fold_source.read_bytes()).hexdigest(),
            'folds': sorted(map(int,d.spatiotemporal_fold.unique())), 'samples':len(d),'positives':int(d.true_label.sum()),
            'scope':'Historical cross-fitting of calibration layer; not a new independent test of the base model.',
            'policy':'Original model, ranking scores and operational risk thresholds unchanged. No automatic production deployment.',
            'limitations':'Existing OOF models may share training data across folds. Calibration method comparison is exploratory; future independent data required.',
            'display_method':'platt', 'metrics':rows}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print('folds:',manifest['folds'],'samples:',len(d),'positives:',manifest['positives'])
