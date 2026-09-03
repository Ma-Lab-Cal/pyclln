"""Default configuration for the reduced-vocab physics language model.
Sets the corpus/vocab knobs so the 4-shape grammar (13,664 edges; 122-token vocabulary plus <BOS>) is defined
identically on import. Imported by clm/__init__.py before any submodule."""
import os
_CFG = dict(
    LM_V2="1", LM_VOCAB_SIZE="132", LM_SIMPLE="1", LM_PHYS="2", LM_DISJOINT_SUBJ="1",
    LM_FAN_OF="2", LM_FAN_LOC="5", LM_FAN_LOCM="4", LM_OF_THE="1", LM_LOC_THE="1",
    LM_MORE_DETS="1", LM_BALANCE_AUX="1",
    LM_DROP_TOKENS="why,can,measure,pure,mixed,coherent,incoherent,classical,quantum,degenerate,excited",
    LM_FAMSET="s2_measure_of,s2_has_loc,s2_shows_loc,s2_shows,q_what_simple",
    LM_VG_CLIP_LO="0.4", LM_FAMILIES="stmt", LM_NO_OBJ_DET="1",
)
for _k, _v in _CFG.items():
    os.environ.setdefault(_k, _v)
