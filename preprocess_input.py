import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def preprocess_input(data, SC):
    """Preprocesses input data for prediction."""

    df = pd.DataFrame([data])  # Create DataFrame from input data

    df["A1Cresult"] = df["A1Cresult"].replace({">7": 2, ">8": 2, "Norm": 1, "None": 0})
    df["max_glu_serum"] = df["max_glu_serum"].replace({">200": 2, ">300": 2, "Norm": 1, "None": 0})
    df.change = df.change.replace("Ch", "Yes")



    mapped_adm = {"Not Available":np.nan}
    df.admission_source_id = df.admission_source_id.replace(mapped_adm)

    mapped_discharge = {"Not Available":"Other"}

    df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(mapped_discharge)

    mapped = {"Not Available":np.nan}

    df.admission_type_id = df.admission_type_id.replace(mapped)

    df.age = df.age.replace({"[70-80)": 75,
                             "[60-70)": 65,
                             "[50-60)": 55,
                             "[80-90)": 85,
                             "[40-50)": 45,
                             "[30-40)": 35,
                             "[90-100)": 95,
                             "[20-30)": 25,
                             "[10-20)": 15,
                             "[0-10)": 5})

    df = df.fillna(0).astype(int, errors='ignore')
    categorical_df = df.select_dtypes('O')

    numerical_df = df.select_dtypes(np.number)

    for i in categorical_df:
        categorical_df[i] = le.fit_transform(categorical_df[i])

    df = pd.concat([numerical_df, categorical_df], axis=1)

    df_scaled = pd.DataFrame(SC.transform(df), columns=df.columns)
    return df_scaled