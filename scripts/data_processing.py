import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

CAT_COLS = ['job', 'city', 'state', 'category', 'merchant']
DROP_COLS = ['trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'trans_num']
ONEHOT_COLS = ['gender']


def build_global_freq_maps(csv_path: Path, cat_cols, chunksize=200_000):
    counters = {c: Counter() for c in cat_cols}

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False, usecols=lambda c: c in cat_cols):
        for c in cat_cols:
            if c in chunk.columns:
                # Count including NaN consistently
                # Convert NaN to a sentinel so it can be counted/mapped
                values = chunk[c].astype("object").where(chunk[c].notna(), "__MISSING__")
                counters[c].update(values.tolist())

    # Convert Counter -> dict for fast mapping
    freq_maps = {c: dict(counters[c]) for c in cat_cols}
    return freq_maps

def preprocess(df: pd.DataFrame, freq_maps: dict) -> pd.DataFrame:
    # dropping columns

    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True) #drop unnamed columns
    drop_cols = [i for i in DROP_COLS if i in df.columns] #drop specfied columns if they exist
    df = df.drop(columns=drop_cols)

    # Converting dob to age and dropping dob
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    if 'trans_date_trans_time' in df.columns:
        asof = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    else:
        asof = pd.Timestamp.today()

    df['age'] = ((asof - df['dob']).dt.days // 365).astype("Int16")
    df = df.drop(columns=['dob'], errors="ignore")

    # frequency encoding for specified columns
    for c in CAT_COLS:
        if c in df.columns:
            values = df[c].astype("object").where(df[c].notna(), "__MISSING__")
            mapped = values.map(freq_maps[c]).fillna(0).astype("int32")
            df[c + "_freq"] = mapped
            df = df.drop(columns=[c])

    # One-hot encoding for specified columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[ONEHOT_COLS])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(ONEHOT_COLS), index=df.index)
    df = pd.concat([df.drop(columns=ONEHOT_COLS), encoded_df], axis=1) #drop original onehot cols and concat the new encoded df
    
    # Downcast remaining numeric types
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")

    # reorder so that is_fraud is the last column for visual convenience
    if 'is_fraud' in df.columns:
        cols = [col for col in df.columns if col != 'is_fraud'] + ['is_fraud']
        df = df[cols]
    
    return df

def process_in_chunks(csv_path: Path, out_path: Path, freq_maps: dict, chunksize: int = 200_000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = True

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk = preprocess(chunk, freq_maps)
        chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False

def main():
    base_dir = Path(__file__).resolve().parent.parent

    csv_path_train = base_dir / "raw_data" / "fraudTrain.csv"
    csv_path_test  = base_dir / "raw_data" / "fraudTest.csv"

    out_path_train = base_dir / "processed_data" / "fraudTrain_processed.csv"
    out_path_test  = base_dir / "processed_data" / "fraudTest_processed.csv"

    print("Building global frequency maps from training data...")
    freq_maps = build_global_freq_maps(csv_path_train, CAT_COLS)

    print("Processing training data...")
    process_in_chunks(csv_path_train, out_path_train, freq_maps)
    print("Saved training processed CSV.")

    print("Processing test data...")
    process_in_chunks(csv_path_test, out_path_test, freq_maps)
    print("Saved test processed CSV.")



if __name__ == "__main__":
    main()