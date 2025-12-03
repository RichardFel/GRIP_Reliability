
# %%
'''
Created by Richard Felius

Main script for the GRIP HAP project.
- Run this script to process raw data
- Get activity scores for every 10 seconds / minute
- creates an excel with outcomes per person per measurement (day)


'''

import logging
from SRC.process_data import process_data
from SRC.reliability import reliability
import os
from datetime import date

# General project settings.
# More specific settings can be found in the config_file.
settings = {
    'PAIN_SCORES': True,
    'RESULTS_2HOURS': False,
    'VERBOSE': True,
    'VISUALISE': False,
    'DATA_DIR': 'Data',
    'MODELS_DIR':'Models',
    'RESULTS_DIR':'Results',
    'FIGURES_DIR':'Figures',
    'frequency': 12.5, # Hz
    'sample_size': 10, # Time in seconds  -> 12 seconds?
    'chunk_size': 6, # Number of 10 seconds epochs per bout
    'MIN_Duration': 600, # Minimum duration of wear time in minutes
    'MAX_Duration': 1200, # Maximum duration of wear time in minutes
    'number_of_days': 6 # Number of days to include in reliability analysis
}
for folder in ["Logging", "Figures", "Results"]:
    os.makedirs(folder, exist_ok=True)
logging.basicConfig(filename=f'logging/warnings_{date.today()}.log', level=logging.WARN)

# %%


def main():
    # Loads the raw data, transform it into 10 seconds epochs and predict activities
    # based on a previously trained ML algorithm.
    # Next, it calculates characteristics based on the predicted activities.
    # Finally, it saves the results in an excel file.
    process_data(settings)
    
    # Calculate reliability 
    reliability(settings)
    
if __name__ == '__main__':
    main()

# %%


#            ,^_^,
#            (  '_}
#            ( ( )  
#    __------( ( )
#  ()(         ( ) 
#  ()(  )_____(  )
#     | |     | |
#     | |     | |
# ---------------------------