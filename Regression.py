import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os

data = pd.read_csv(r"C:\Users\Victor Steinrud\Downloads\fraudTrain.csv")

data = data.drop(["trans_date_trans_time", "first", "last", "street", "trans_num", "unix_time", "merchant", "cc_num"], axis=1)

cat_encoder = OneHotEncoder()

#job 1hot
def read_txt_to_list(filename):
    with open(filename, 'r') as file:     
        return [line.strip() for line in file.read().split(';')]

os.chdir(r"C:\Users\Victor Steinrud\Documents\DAKI\P1\Feature-valg\jobgroups")
txt_files = [file for file in os.listdir() if file.endswith('.txt')]
txt_data = {}
for file in txt_files:
    txt_data[file.replace('.txt', '')] = read_txt_to_list(file)

for index, row in data.iterrows():
    for category, values in txt_data.items():
        if row['job'] in values:  
            data.at[index, 'job'] = category
            break

job_cat = data[['job']]
job_cat_1hot = cat_encoder.fit_transform(job_cat)
job_cat_1hot_array = job_cat_1hot.toarray()

#job fordelt i features
job_cat_df = pd.DataFrame(job_cat_1hot_array)

# Rename the columns to the desired names
job_cat_df.columns = cat_encoder.categories_[0]

# Concatenate with the main dataframe
data = pd.concat([data, job_cat_df], axis=1)

# Drop the old 'job' column
data = data.drop('job', axis=1)




#category 1hot
Purchase_category = data[["category"]]
purchase_cat_1hot = cat_encoder.fit_transform(Purchase_category)
purchase_cat_1hot_array = purchase_cat_1hot.toarray()


#category fordelt i features
cat_df = pd.DataFrame(purchase_cat_1hot_array)

# Rename the columns to the desired names
cat_df.columns = cat_encoder.categories_[0]

# Concatenate with the main dataframe
data = pd.concat([data, cat_df], axis=1)

# Drop the old 'job' column
data = data.drop('category', axis=1)

# print(data.head())

#gender 1hot
gender = data[["gender"]]

data["gender"] = data["gender"].replace({"M": 1, "F": 0})



gender_1hot = cat_encoder.fit_transform(gender)
gender_1hot_array = gender_1hot.toarray()


#dob 1hot
data['dob'] = data['dob'].str[2].map({
    '0': '1900s',
    '1': '1910s',
    '2': '1920s',
    '3': '1930s',
    '4': '1940s',
    '5': '1950s',
    '6': '1960s',
    '7': '1970s',
    '8': '1980s',
    '9': '1990s'
})

dob = data[["dob"]]
dob_1hot = cat_encoder.fit_transform(dob)
dob_1hot_array = dob_1hot.toarray()


dob_df = pd.DataFrame(dob_1hot_array)

# Rename the columns to the desired names
dob_df.columns = cat_encoder.categories_[0]

# Concatenate with the main dataframe
data = pd.concat([data, dob_df], axis=1)

# Drop the old 'job' column
data = data.drop('dob', axis=1)
#zip skal 1hot eller noget andet


#state 1hot
state = data[["state"]]
state_1hot = cat_encoder.fit_transform(state)
state_1hot_array = state_1hot.toarray()



state_df = pd.DataFrame(state_1hot_array)

# Rename the columns to the desired names
state_df.columns = cat_encoder.categories_[0]

# Concatenate with the main dataframe
data = pd.concat([data, state_df], axis=1)

# Drop the old 'job' column
data = data.drop('state', axis=1)

# print(data.head())

#features der skal standardiseres er: 
    #amt, lat, long, city_pop, merch_lat, merch_long, 
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

#standardiseret amt
amt_values = data['amt']
amt_array = amt_values.values.reshape(-1, 1)
amt_std_scaled = std_scaler.fit_transform(amt_array)

#standardiseret lat
lat_values = data['lat']
lat_array = lat_values.values.reshape(-1, 1)
lat_std_scaled = std_scaler.fit_transform(lat_array)

#standardiseret long
long_values = data['long']
long_array = long_values.values.reshape(-1, 1)
long_std_scaled = std_scaler.fit_transform(long_array)

#standardiseret citypop
citypop_values = data['city_pop']
citypop_array = citypop_values.values.reshape(-1, 1)
citypop_std_scaled = std_scaler.fit_transform(citypop_array)

#standardiseret merch_lat
mlat_values = data['merch_lat']
mlat_array = mlat_values.values.reshape(-1, 1)
mlat_std_scaled = std_scaler.fit_transform(mlat_array)

#standardiseret merch_long
mlong_values = data['merch_long']
mlong_array = mlong_values.values.reshape(-1, 1)
mlong_std_scaled = std_scaler.fit_transform(mlong_array)





medical_health_array = data['MedicalnHealth'].values.reshape(-1, 1)
education_training_array = data['EducationnTraining'].values.reshape(-1, 1)
environment_nature_array = data['EnvironmentnNature'].values.reshape(-1, 1)
law_governance_array = data['LawnGovernance'].values.reshape(-1, 1)
design_media_array = data['DesignnMedia'].values.reshape(-1, 1)
other_array = data['Other'].values.reshape(-1, 1)
business_management_array = data['BusinessnManegement'].values.reshape(-1, 1)
engineering_it_array = data['EngineeringnIT'].values.reshape(-1, 1)


misc_net_array = data['misc_net'].values.reshape(-1, 1)
grocery_pos_array = data['grocery_pos'].values.reshape(-1, 1)
entertainment_array = data['entertainment'].values.reshape(-1, 1)
gas_transport_array = data['gas_transport'].values.reshape(-1, 1)
misc_pos_array = data['misc_pos'].values.reshape(-1, 1)
grocery_net_array = data['grocery_net'].values.reshape(-1, 1)
shopping_net_array = data['shopping_net'].values.reshape(-1, 1)
shopping_pos_array = data['shopping_pos'].values.reshape(-1, 1)
food_dining_array = data['food_dining'].values.reshape(-1, 1)
personal_care_array = data['personal_care'].values.reshape(-1, 1)
health_fitness_array = data['health_fitness'].values.reshape(-1, 1)
travel_array = data['travel'].values.reshape(-1, 1)
kids_pets_array = data['kids_pets'].values.reshape(-1, 1)
home_array = data['home'].values.reshape(-1, 1)


decade_1900s_array = data["1900s"].values.reshape(-1, 1)
decade_1920s_array = data["1920s"].values.reshape(-1, 1)
decade_1930s_array = data["1930s"].values.reshape(-1, 1)
decade_1940s_array = data["1940s"].values.reshape(-1, 1)
decade_1950s_array = data["1950s"].values.reshape(-1, 1)
decade_1960s_array = data["1960s"].values.reshape(-1, 1)
decade_1970s_array = data["1970s"].values.reshape(-1, 1)
decade_1980s_array = data["1980s"].values.reshape(-1, 1)
decade_1990s_array = data["1990s"].values.reshape(-1, 1)



NC_array = data["NC"].values.reshape(-1, 1)
WA_array = data["WA"].values.reshape(-1, 1)
ID_array = data["ID"].values.reshape(-1, 1)
MT_array = data["MT"].values.reshape(-1, 1)
VA_array = data["VA"].values.reshape(-1, 1)
PA_array = data["PA"].values.reshape(-1, 1)
KS_array = data["KS"].values.reshape(-1, 1)
TN_array = data["TN"].values.reshape(-1, 1)
IA_array = data["IA"].values.reshape(-1, 1)
WV_array = data["WV"].values.reshape(-1, 1)
FL_array = data["FL"].values.reshape(-1, 1)
CA_array = data["CA"].values.reshape(-1, 1)
NM_array = data["NM"].values.reshape(-1, 1)
NJ_array = data["NJ"].values.reshape(-1, 1)
OK_array = data["OK"].values.reshape(-1, 1)
IN_array = data["IN"].values.reshape(-1, 1)
MA_array = data["MA"].values.reshape(-1, 1)
TX_array = data["TX"].values.reshape(-1, 1)
WI_array = data["WI"].values.reshape(-1, 1)
MI_array = data["MI"].values.reshape(-1, 1)
WY_array = data["WY"].values.reshape(-1, 1)
HI_array = data["HI"].values.reshape(-1, 1)
NE_array = data["NE"].values.reshape(-1, 1)
OR_array = data["OR"].values.reshape(-1, 1)
LA_array = data["LA"].values.reshape(-1, 1)
DC_array = data["DC"].values.reshape(-1, 1)
KY_array = data["KY"].values.reshape(-1, 1)
NY_array = data["NY"].values.reshape(-1, 1)
MS_array = data["MS"].values.reshape(-1, 1)
UT_array = data["UT"].values.reshape(-1, 1)
AL_array = data["AL"].values.reshape(-1, 1)
AR_array = data["AR"].values.reshape(-1, 1)
MD_array = data["MD"].values.reshape(-1, 1)
GA_array = data["GA"].values.reshape(-1, 1)
ME_array = data["ME"].values.reshape(-1, 1)
AZ_array = data["AZ"].values.reshape(-1, 1)
MN_array = data["MN"].values.reshape(-1, 1)
OH_array = data["OH"].values.reshape(-1, 1)
CO_array = data["CO"].values.reshape(-1, 1)
VT_array = data["VT"].values.reshape(-1, 1)
MO_array = data["MO"].values.reshape(-1, 1)
SC_array = data["SC"].values.reshape(-1, 1)
NV_array = data["NV"].values.reshape(-1, 1)
IL_array = data["IL"].values.reshape(-1, 1)
NH_array = data["NH"].values.reshape(-1, 1)
SD_array = data["SD"].values.reshape(-1, 1)
AK_array = data["AK"].values.reshape(-1, 1)
ND_array = data["ND"].values.reshape(-1, 1)
CT_array = data["CT"].values.reshape(-1, 1)
RI_array = data["RI"].values.reshape(-1, 1)
DE_array = data["DE"].values.reshape(-1, 1)

is_fraud_array = data["is_fraud"].values.reshape(-1, 1)

# 1. Combine all features
all_features = np.hstack([
    is_fraud_array,

    medical_health_array,
    education_training_array,
    environment_nature_array,
    law_governance_array,
    design_media_array,
    other_array,
    business_management_array,
    engineering_it_array,
    misc_net_array,
    grocery_pos_array,
    entertainment_array,
    gas_transport_array,
    misc_pos_array,
    grocery_net_array,
    shopping_net_array,
    shopping_pos_array,
    food_dining_array,
    personal_care_array,
    health_fitness_array,
    travel_array,
    kids_pets_array,
    home_array,

    amt_std_scaled,
    lat_std_scaled,
    long_std_scaled,
    citypop_std_scaled,
    mlat_std_scaled,
    mlong_std_scaled,

    decade_1900s_array,
    decade_1920s_array,
    decade_1930s_array,
    decade_1940s_array,
    decade_1950s_array,
    decade_1960s_array,
    decade_1970s_array,
    decade_1980s_array,
    decade_1990s_array,

    NC_array,
    WA_array,
    ID_array,
    MT_array,
    VA_array,
    PA_array,
    KS_array,
    TN_array,
    IA_array,
    WV_array,
    FL_array,
    CA_array,
    NM_array,
    NJ_array,
    OK_array,
    IN_array,
    MA_array,
    TX_array,
    WI_array,
    MI_array,
    WY_array,
    HI_array,
    NE_array,
    OR_array,
    LA_array,
    DC_array,
    KY_array,
    NY_array,
    MS_array,
    UT_array,
    AL_array,
    AR_array,
    MD_array,
    GA_array,
    ME_array,
    AZ_array,
    MN_array,
    OH_array,
    CO_array,
    VT_array,
    MO_array,
    SC_array,
    NV_array,
    IL_array,
    NH_array,
    SD_array,
    AK_array,
    ND_array,
    CT_array,
    RI_array,
    DE_array


])

# 2. Compute the covariance matrix
cov_matrix = np.cov(all_features, rowvar=False)  # rowvar=False to compute covariance between columns

import seaborn as sns
correlation_matrix = np.corrcoef(cov_matrix, rowvar=False)



# Generate labels for the axes
    # job_labels = ['job_' + cat for cat in cat_encoder.categories_[0]]
    # purchase_labels = ['category_' + cat for cat in cat_encoder.categories_[0]]
    # gender_labels = ['gender_' + cat for cat in cat_encoder.categories_[0]]
    # dob_labels = ['dob_' + cat for cat in cat_encoder.categories_[0]]
    # state_labels = ['state_' + cat for cat in cat_encoder.categories_[0]]

    # # Combine all labels
labels = [ 
        'Is_Fraud',

        'MedicalnHealth',
        'EducationnTraining',
        'EnvironmentnNature',
        'LawnGovernance',
        'DesignnMedia',
        'Other',
        'BusinessnManegement',
        'EngineeringnIT'

        'amt_std_scaled',
        'lat_std_scaled',
        'long_std_scaled',
        'citypop_std_scaled',
        'mlat_std_scaled',
        'mlong_std_scaled'
        
        'misc_net_trans',
        'grocery_pos_trans',
        'entertainment_trans',
        'gas_transport_trans',
        'misc_pos_trans',
        'grocery_net_trans',
        'shopping_net_trans',
        'shopping_pos_trans',
        'food_dining_trans',
        'personal_care_trans',
        'health_fitness_trans',
        'travel_trans',
        'kids_pets_trans',
        'home_trans',

        'dob_1900s',
        'dob_1920s',
        'dob_1930s',
        'dob_1940s',
        'dob_1950s',
        'dob_1960s',
        'dob_1970s',
        'dob_1980s',
        'dob_1990s',
        'NC',
        'WA',
        'ID',
        'MT',
        'VA',
        'PA',
        'KS',
        'TN',
        'IA',
        'WV',
        'FL',
        'CA',
        'NM',
        'NJ',
        'OK',
        'IN',
        'MA',
        'TX',
        'WI',
        'MI',
        'WY',
        'HI',
        'NE',
        'OR',
        'LA',
        'DC',
        'KY',
        'NY',
        'MS',
        'UT',
        'AL',
        'AR',
        'MD',
        'GA',
        'ME',
        'AZ',
        'MN',
        'OH',
        'CO',
        'VT',
        'MO',
        'SC',
        'NV',
        'IL',
        'NH',
        'SD',
        'AK',
        'ND',
        'CT',
        'RI',
        'DE'


    ]


    # threshold = 0.7
    # mask = abs(correlation_matrix) > threshold






    # plt.figure(figsize=(24, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, mask=~mask, labels=labels )
# sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, mask=mask, annot=False, labels=labels)





plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels)
# plt.title('Correlation Matrix Heatmap')
# plt.xlabel('Features')
# plt.ylabel('Features')



plt.show()