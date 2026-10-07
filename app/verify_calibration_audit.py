"""Check calibration math and execute the UI block with a minimal Streamlit stub."""
from pathlib import Path
import numpy as np
import pandas as pd
from calibration_tools import isotonic, platt, metrics

root = Path(__file__).resolve().parents[1]
x = np.array([.1,.1,.2,.3,.4,.4,.7,.8])
y = np.array([0,1,0,0,1,0,1,1])
grid = np.linspace(0,1,100)
for fn in [isotonic,platt]:
    p = fn(x,y,grid)
    assert np.all(np.diff(p)>=-1e-10) and np.all((p>=0)&(p<=1))
    assert np.allclose(p,fn(x[::-1],y[::-1],grid))
d = pd.read_csv(root/'outputs/calibration_audit/crossfit_predictions.csv')
for fn, col in [(platt,'platt_crossfit_prob'),(isotonic,'isotonic_crossfit_prob')]:
    for fold in d.spatiotemporal_fold.unique():
        mask = d.spatiotemporal_fold.eq(fold)
        check = fn(d.loc[~mask,'ensemble_prob'],d.loc[~mask,'true_label'],d.loc[mask,'ensemble_prob'])
        assert np.allclose(check,d.loc[mask,col])
comparison = pd.read_csv(root/'outputs/calibration_audit/comparison.csv').set_index('method')
for method,col in [('uncalibrated','ensemble_prob'),('platt','platt_crossfit_prob'),('isotonic','isotonic_crossfit_prob')]:
    for key,val in metrics(d.true_label,d[col]).items():
        assert np.isclose(val,comparison.loc[method,key])

class Stub:
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def expander(self,*args,**kwargs): return self
    def columns(self,n): return [self]*n
    def pyplot(self,fig,**kwargs):
        pass
    def __getattr__(self,name): return lambda *args,**kwargs: None

source = (root/'app/app.py').read_text(encoding='utf-8')
start = source.index("    with st.expander('🎯 风险概率校准")
end = source.index("    with st.expander('⚠️ 今日行动建议",start)
import textwrap
class PlotStub(Stub):
    def subplots(self,**kwargs): return self,self
exec(textwrap.dedent(source[start:end]), {'st':Stub(),'OUTPUTS':root/'outputs','pd':pd,'np':np,'plt':PlotStub()})
print('PASS: monotonicity, ties, fold separation, metrics and calibration UI block')
