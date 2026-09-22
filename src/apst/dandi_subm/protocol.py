"""Frozen sub-M 2015 CO source/dev/final contract."""
from pathlib import Path
import hashlib, json, os


def output_root(root: str | Path | None = None) -> Path:
 """Return the campaign-artifact directory without depending on a checkout."""
 value = root if root is not None else os.environ.get("APST_SUBM_ROOT", "dandi_subm")
 return Path(value).expanduser().resolve()


def raw_root(root: str | Path | None = None) -> Path:
 """Return explicit raw-data storage or the portable DANDI sub-M default."""
 value = root if root is not None else os.environ.get("APST_DATA_ROOT", "data/000688/sub-M")
 return Path(value).expanduser().resolve()


DEFAULT_RAW_ROOT = raw_root()
BIN_SECONDS=.020; WINDOW_BINS=50; ACTIVITY_TRIALS=33; CARRIER_TRIALS=33; Q50_TRIAL_INDEX=50; MAX_UNITS=100; CANONICAL_ELECTRODES=96; MOVE_START_AFTER_GO_SECONDS=.1; MOVE_STOP_AFTER_GO_SECONDS=.6
TRAIN_SESSIONS=('sub-M_ses-CO-20150511','sub-M_ses-CO-20150512','sub-M_ses-CO-20150610','sub-M_ses-CO-20150611','sub-M_ses-CO-20150612','sub-M_ses-CO-20150615')
DEV_SESSIONS=('sub-M_ses-CO-20150616','sub-M_ses-CO-20150617')
FINAL_SESSIONS=('sub-M_ses-CO-20150623','sub-M_ses-CO-20150625','sub-M_ses-CO-20150626')
def split_for(s):
 for k,v in [('train',TRAIN_SESSIONS),('dev',DEV_SESSIONS),('final',FINAL_SESSIONS)]:
  if s in v:return k
 raise ValueError('outside frozen sub-M 2015 roster: '+s)
def assert_authorized(s,purpose):
 if purpose not in ('source','development'):raise PermissionError('final access requires sealed final gate')
 split=split_for(s)
 if (purpose=='source' and split!='train') or (purpose=='development' and split!='dev'):raise PermissionError('unauthorized '+purpose+' access: '+s)
 return split
def protocol_dict(): return {'schema':'dandi688_subm_2015_co_v1','train_sessions':list(TRAIN_SESSIONS),'dev_sessions':list(DEV_SESSIONS),'final_sessions':list(FINAL_SESSIONS),'bin_seconds':BIN_SECONDS,'window_bins':WINDOW_BINS,'activity_trials':ACTIVITY_TRIALS,'carrier_trials':CARRIER_TRIALS,'q50_trial_index':Q50_TRIAL_INDEX,'max_units':MAX_UNITS,'canonical_electrodes':CANONICAL_ELECTRODES,'move_window_after_go_seconds':[.1,.6],'site':'M1_only','representation':'sua_only'}
PROTOCOL_SHA256=hashlib.sha256(json.dumps(protocol_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
