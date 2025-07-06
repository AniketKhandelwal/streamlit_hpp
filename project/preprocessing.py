import pandas as pd

def categorical_encoding(dataframe):
    ordinal_mappings = {
        'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
        'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
        'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'BsmtQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'BsmtCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3},
        'BsmtExposure': {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3},
        'BsmtFinType1': {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
        'BsmtFinType2': {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5},
        'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
        'GarageFinish': {'Unf': 0, 'RFn': 1, 'Fin': 2},
        'GarageQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'GarageCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
        'PavedDrive': {'N': 0, 'P': 1, 'Y': 2}
    }

    ordinal_cols = list(ordinal_mappings.keys())

    nominal_cols = [
        'MSZoning', 'Street', 'LandContour', 'Utilities', 'LotConfig',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation',
        'Heating', 'CentralAir', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition'
    ]

    for col in ordinal_cols:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].map(ordinal_mappings[col])

    encoded_data = pd.get_dummies(
        dataframe,
        columns=[col for col in nominal_cols if col in dataframe.columns],
        drop_first=True,
        dtype='int'
    )

    return encoded_data
