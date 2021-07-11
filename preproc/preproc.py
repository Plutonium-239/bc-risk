import pandas as pd

df = pd.read_excel('../data/Dataset_model1.xlsx', engine='openpyxl')
# nalist = df.columns[df.isna().any()]
nalist = ['No of Children', 'Gender of children', 'age at 1st child (yrs)', 
'Syndrome', # 'gene', 'gene name if VUS',
'family testing done',
'family members tested', 'members positive', 'IES-R Score',
'FinalDass21Score', 'OCP use', 'tubal ligation']

replacewith = {a:df[a].mode()[0] for a in nalist}
replacewith['Gender of children'] = 0
replacewith['No of Children'] = 0
replacewith['Syndrome'] = 0
# replacewith['gene'] = None
# replacewith['gene name if VUS'] = None
replacewith['family testing done'] = 0
replacewith['family members tested'] = 0
replacewith['members positive'] = 0
replacewith['tubal ligation'] = 0
# print(replacewith)

df['outcome'] = 0
# df.outcome.loc[pd.notna(df['gene'])] = 1
df['male children'] = 0
df['female children'] = 0
for i in df.index:
	row = df.loc[i]
	if pd.notna(df.loc[i,'gene']):
		df.loc[i,'outcome'] = 1
	if df.loc[i,'No of Children'] != 0:
		gender = str(df.loc[i,'Gender of children'])
		df.loc[i,'male children'] = gender.count('1')
		df.loc[i,'female children'] = gender.count('2')
	elif df.loc[i,'No of Children'] == 0:
		df.loc[i,'Gender of children'] = 0

for col in nalist:
	# mode = df[col].mode()[0]
	df[col].fillna(value=replacewith[col], inplace=True)

df = df.set_index('Sno')
print(df.isna().any())





df.to_csv('../data/model1_preprocd.csv')

