#EDA with Pandas, assume it will highlight further exploration

import pandas as pd
from pandas_profiling import ProfileReport

#individual break down
import pandas as pd
df = pd.read_csv('ds_salaries.csv', index_col=0)
print('3 RANDOM ROWS\n')
print(df.sample(3))
print('NUMBER OF ROWS AND COLUMNS')
print(df.shape)
print('NUMBER OF MISSING VALUES')
print(df.isnull().sum().sum())
print('TOTAL NUMBER OF VALUES')
print(df.size)
print('COLUMN NAMES AND DATA TYPES')
print(df.dtypes)
print('COMPANY SIZES')
print(df['company_size'].unique())
print('NUMERIC DATA STATS')
print(df.describe())
print('NUMBER OF UNIQUE JOB TITLES')
print(df['job_title'].nunique())
print('EMPLOYMENT TYPE (FRACTION)')
print(df['employment_type'].value_counts(normalize=True))
print('THE 3 LARGEST SALARIES (IN USD)')
print(df['salary_in_usd'].nlargest(3))
print('NUMERIC COLUMN CORRELATION')
print(df.corr())
print('SALARY IN USD VS. SALARY IN LOCAL CURRENCY\n')
print(df.plot(x='salary', y='salary_in_usd', kind='scatter'))
print('SALARY IN USD\n')
print(df['salary_in_usd'].plot(kind='hist'))




#profile
df = pd.read_csv('data/NOPIMS_Australia/Ironbank-1.csv', na_values=-999)
report = ProfileReport(df)
report