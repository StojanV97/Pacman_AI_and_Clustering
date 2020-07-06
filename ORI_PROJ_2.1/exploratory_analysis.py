
import pandas as pd
from pandas_profiling import ProfileReport

credit_cards = pd.read_csv('credit_card_data.csv')
profile = ProfileReport(credit_cards, title="Pandas Profiling Report")
profile.to_file(output_file='C:\\Users\\Stoja\\Desktop\\exploratory_analysis.html')
