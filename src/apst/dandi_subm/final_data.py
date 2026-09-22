"""Final-only loader adapter; keeps the training import closure frozen."""
from pathlib import Path
from apst.dandi_subm import final_access, subm_data
def load_final_pair(session_id,cap):
 return subm_data.load_pair(session_id,purpose='final',final_access=cap)

def load_final_cached(path,cap):
 return subm_data.load_cached_session(Path(path),final_access=cap)
