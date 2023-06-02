import pandas as pd
import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
from tqdm import tqdm

def generate_fp_for_smiles(smiles):
    try:
        ms = Chem.MolFromSmiles(smiles)
        if ms is None:
            print("Invalid SMILES")
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(ms, 2, nBits=2048)
        fp_arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
        for i in range(fp.GetNumBits()):
            if fp.GetBit(i):
                fp_arr[i] = 1
        fp_reshaped_array = fp_arr.reshape(1, 2048)
        return fp_reshaped_array
    except Exception as e:
        print(f"Error generating fingerprint for SMILES: {smiles}")
        print(str(e))
        return None

def generate_fp(row):
    index, data = row
    smi = data['smiles']
    class_1row = data['Class']
    try:
        fp_reshaped_array = generate_fp_for_smiles(smi)
        if fp_reshaped_array is not None:
            result =  [smi, class_1row] + fp_reshaped_array.tolist()[0]
            return '\t'.join(map(str, result))
        else:
            return None
    except Exception as e:
        print(f"Error generating fingerprint for SMILES: {smi}")
        print(str(e))
        return None

if __name__ == '__main__':
    print("entered in to main")
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    print("getting_size")
    workers = 16
    reader = pd.read_csv(input_filename)
    columns = ['smiles', 'Class'] + [f'fp{i}' for i in range(2048)]
    with open(output_filename,'w') as f:
        f.write('\t'.join(columns) + '\n')
        with Pool(workers) as pool:
            #result = list(tqdm(pool.imap(generate_fp, reader.iterrows(), chunksize=100), total=len(reader)))
            #df_fps = df_fps.append(result)
            for result in tqdm(pool.imap(generate_fp,reader.iterrows(),chunksize=100),total=len(reader)):
                if result is not None:
                    f.write(result + '\n')
    print(f"Processed {len(reader)} molecules.")
