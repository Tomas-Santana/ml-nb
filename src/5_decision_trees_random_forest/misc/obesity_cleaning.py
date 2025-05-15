import pandas as pd

def label_encode(value: str, mapping):
    return mapping[value.lower()]
  
CAEC_CALC_mapping = {
  "no": 1,
  "sometimes": 2,
  "frequently": 3,
  "always": 4
}

yes_no_mapping = {
  "no": 0,
  "yes": 1
}
  
  

df = pd.read_csv("./datasets/obesity_raw.csv")

# make gender column binary
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 0)


# make family history with overweight binary
df["family_history_with_overweight"] = df["family_history_with_overweight"].apply(lambda x: label_encode(x, yes_no_mapping))
df["FAVC"] = df["FAVC"].apply(lambda x: label_encode(x, yes_no_mapping))
df["SMOKE"] = df["SMOKE"].apply(lambda x: label_encode(x, yes_no_mapping))
df["SCC"] = df["SCC"].apply(lambda x: label_encode(x, yes_no_mapping))

df["CAEC"] = df["CAEC"].apply(lambda x: label_encode(x, CAEC_CALC_mapping))
df["CALC"] = df["CALC"].apply(lambda x: label_encode(x, CAEC_CALC_mapping))

# make FCVC int
df["FCVC"] = df["FCVC"].astype(int)
df['CH2O'] = df['CH2O'].astype(int)
df["FAF"] = df["FAF"].astype(int)
df["TUE"] = df["TUE"].astype(int)

df.drop(columns=["MTRANS"], inplace=True)

df.to_csv("./datasets/obesity_clean.csv", index=False)

