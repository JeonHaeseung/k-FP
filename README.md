# k-fingerprinting (k-FP)
This repository contains the Python 3 implementation of the **k-FP** WF attack as described in the [USENIX Security '16 paper](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_hayes.pdf).


## Major Changes
1. Converted the code from **Python 2** to **Python 3**:
   - Updated format and syntax to be compatible with Python 3.
2. Changed dataset format:
   - **Old format**: `.txt` file.
   - **New format**: `.npz` file.

   Ensure that you have a **Tik-Tok** format `.npz` dataset (`direction*timestamp`) in the `/data` folder before running the code. 

## Modifying for Other Data Formats
If your dataset is in a different format:
- Update the `'dictionary_'` method in the `k-FP.py` file.
- Update the `get_pkt_list` method in the `RF_fextract.py` file