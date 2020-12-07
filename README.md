# LIET

---Status---

12/7/20: Currently the LIET class is fully functional, but is only operable in 
the context of a jupyter notebook (rnap_lib_LIET_class.py). The data sim lib 
contains all necessary functions to simulate data from the full model, and so 
can be used for unit testing (rnap_lib_data_sim.py). The high-throughput input 
data processing lib is bare bones at present (rnap_lib_data_proc.py)---
requires a closed non-overlaped annotation file and the only options for 
bedgraph file parsing are fixed size (in bases) padding or fractional (as a 
percentage of the annotation length) padding. The padding is equal sized on 
both 5' and 3' ends of annotation. The high-throughput executable is still only 
about 30% of the way to alpha (rnap_LIET_exe.py).