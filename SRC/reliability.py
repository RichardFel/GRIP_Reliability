# %%
#
import pandas as pd
import pingouin as pg
import numpy as np

#%%
def clean(data, settings):
    # Minimum duration of 10 hours
    print(f'Data points before duration filter: {len(data)}')
    data_pre = len(data)
    data = data.loc[data['Epochs_of_1minute'] >= settings['MIN_Duration']]
    data = data.loc[data['Epochs_of_1minute'] <= settings['MAX_Duration']]
    data_post = len(data)
    print(f'Data points removed due to duration filter: {data_pre - data_post}')
    # characteristics
    total_time = data['Epochs_of_1minute'].sum()  
    print(f'Total wear time in hours: {total_time}')

    print(f'Sedentairy time: {data["sedentairy_count"].sum() / total_time}')
    print(f'Light time: {data["light_count"].sum() / total_time}')
    print(f'Moderate time: {data["moderate_count"].sum() / total_time}')
    
    # exclude_vars = ['sedentairy_perc', 'std_change', 'Sample_entropy_m2_tau1', 'PLZC_dalay1_dim2',
    #             'PLZC_dealy2_dim2', 'PLZC_dealy2_dim3', 'PLZC_dealy3_dim3',
    #             'Sample_entropy_m2_tau1', 'Sample_entropy_m3_tau1', 'Sample_entropy_m4_tau1',
    #             'Sample_entropy_m2_tau2', 'Sample_entropy_m3_tau2', 'Sample_entropy_m4_tau2',
    #             'norm_transitions_1_0', 'norm_transitions_2_1',
    #             'norm_transitions_1_2', 'epochs_moderate_perc',
    #             'moderate_count', 'light_count', 'sedentairy_count',
    #             'sedentairy_median_length','light_median_length',
    #             'moderate_median_length', 'epochs_light_perc']
    # for var in exclude_vars:
    #     if var in data.columns:
    #         data = data.drop(columns=[var])
            
    data = data.dropna(axis=1)
    data = data.sort_values(by=['day'])



    return data

def correlations(data):
    data = data.loc[:, 'per_change':]
    correlations = data.corr(numeric_only=True, method='spearman')
    correlations.dropna(how='all', inplace=True)
    correlations.dropna(axis = 1, how='all', inplace=True)
    correlations *= 100
    correlations.to_excel(f'Results/correlations.xlsx')

    high_corr_vars = []
    for column in correlations.columns:
        for index, value in correlations[column].items():
            if index != column and abs(value) >= 85:
                high_corr_vars.append((index, column, value))
                print(f'High correlation between {index} and {column}: {value}')

def add_weekend_flag(date):
    pd_date = pd.to_datetime(date)
    if pd_date.weekday() < 5:
        return 0
    else:
        return 1

def ICC_analysis(tmp_data, group, week_type, number_of_days, settings):
    subject_counts = tmp_data['subject'].value_counts()


    final_df = pd.DataFrame()
    final_df_excel = pd.DataFrame()
    subjects = subject_counts.loc[subject_counts >= number_of_days * 2].index
    print(subjects)
    for i in range(1, number_of_days+1):
        results = {}
        results_excel = {}
        subjects_data = tmp_data.loc[tmp_data['subject'].isin(subjects)]

        # Get the first 'i' rows for each subject
        subjects_data = subjects_data.set_index('subject')
        subjects_data = subjects_data.loc[:, 'Epochs_of_1minute':]

        # Get the first 'i' rows for each subject and calculate the mean
        first = subjects_data.groupby('subject').head(i)
        mean_values_first = first.groupby('subject').mean()
        # Add a column to indicate 'first' group
        mean_values_first['group_label'] = 1

        second = subjects_data.groupby('subject').apply(lambda x: x.iloc[i:number_of_days+i])
        mean_values_second = second.groupby('subject').mean()
        # Add a column to indicate 'second' group
        mean_values_second['group_label'] = 2

        # Concatenate the mean values of the first and second groups
        final_mean_df = pd.concat(
            [mean_values_first, mean_values_second]).reset_index()
        # print(f' \n Aantal dagen: {i}, aantal proefpersonen {len(final_mean_df) / 2} \n')
        for variable in subjects_data.columns:
            try:
                if ((final_mean_df[variable].mean() == 0 )or np.isnan(final_mean_df[variable].mean())):
                    continue
                icc = pg.intraclass_corr(data=final_mean_df, targets='subject', raters='group_label',
                                        ratings=variable, nan_policy='omit').round(3)
                icc2 = icc.loc[icc['Type'] == 'ICC2']
                CI = icc['CI95%'].loc[1]
                SEM = np.std(final_mean_df[variable], ddof=1) * np.sqrt(1 - icc2['ICC'].values[0])
                MDC = 1.96 * SEM * np.sqrt(2)
                # print(f'{variable} MDC: {MDC}')
                results[f'ICC_{variable}'] = round(float(icc2['ICC'].values[0]), 2)
                results_excel[f'ICC_{variable}'] = [round(float(icc2['ICC'].values[0]), 2), round(float(CI[0]), 2), round(float(CI[1]), 2), round(float(SEM), 2), round(float(MDC), 2), round(float(MDC) / np.std(final_mean_df[variable], ddof=1), 2)]
                # f'{icc2['ICC'].values[0]:.2f} [{CI[0]:.2f}-{CI[1]:.2f}] ({MDC:.2f})'.replace('.', ',')
                # results_excel[f'ICC_{variable}'] = f'{icc2['ICC'].values[0]:.2f} [{CI[0]:.2f}-{CI[1]:.2f}] ({MDC:.2f})'.replace('.', ',')
                # results[f'CI_days_{i}_var_{variable}'] = icc2['CI95%'].values[0]
            except:
                continue
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=[i])
        metric_names = ['ICC', 'CI_low', 'CI_high', 'SEM', 'MDC', 'MDC_pct']
        results_df_excel = pd.DataFrame.from_dict(
            results_excel, orient='index',
            columns=pd.MultiIndex.from_product([[i], metric_names]))
        if final_df.empty:
            final_df = results_df
        if final_df_excel.empty:
            final_df_excel = results_df_excel
        else:
            final_df = pd.concat((final_df, results_df), axis=1)
            final_df_excel = pd.concat((final_df_excel, results_df_excel), axis=1)
    final_df.to_excel(f"Results/ICC_{settings['MIN_Duration']}_{settings['MAX_Duration']}_{group}_{week_type.lower()}_N{len(subjects)}_{number_of_days}days.xlsx")
    final_df_excel.to_excel(f'Results/ICC_{settings['MIN_Duration']}_{settings['MAX_Duration']}_formatted_{group}_{week_type.lower()}_N{len(subjects)}_{number_of_days}days.xlsx')
    subjects_data.corr(numeric_only=True, method='spearman').to_excel(f'Results/correlations_{settings['MIN_Duration']}_{settings['MAX_Duration']}_{group}_{week_type.lower()}_N{len(subjects)}_{number_of_days}days.xlsx')
    
def weekday_weekend_ttest(tmp_data, group, settings):
    variables = tmp_data.loc[:, 'per_change':].select_dtypes(include=np.number).columns
    variables = [v for v in variables if v != 'weekend']

    subject_means = (tmp_data.groupby(['subject', 'weekend'])[variables]
                     .mean()
                     .unstack('weekend'))

    paired = subject_means.dropna(how='any')
    n_pairs = len(paired)
    print(f'Group: {group} — paired subjects (have both weekday & weekend data): {n_pairs}')

    rows = []
    for var in variables:
        weekday_vals = paired[(var, 0)]
        weekend_vals = paired[(var, 1)]
        if weekday_vals.std() == 0 and weekend_vals.std() == 0:
            continue
        result = pg.ttest(weekday_vals, weekend_vals, paired=True).round(4)
        rows.append({
            'variable': var,
            'n_pairs': n_pairs,
            'weekday_mean': weekday_vals.mean(),
            'weekday_std': weekday_vals.std(),
            'weekend_mean': weekend_vals.mean(),
            'weekend_std': weekend_vals.std(),
            'mean_diff': weekday_vals.mean() - weekend_vals.mean(),
            't': result['T'].values[0],
            'dof': result['dof'].values[0],
            'p_value': result['p-val'].values[0],
            'cohen_d': result['cohen-d'].values[0],
            'CI95%': result['CI95%'].values[0],
        })

    results_df = pd.DataFrame(rows).set_index('variable')
    results_df.to_excel(
        f"Results/weekday_vs_weekend_ttest_>{settings['MIN_Duration']}_<{settings['MAX_Duration']}_{group}.xlsx"
    )
    return results_df


def reliability(settings):
    data =  pd.read_excel(f'{settings['RESULTS_DIR']}/results_per_day.xlsx')

    # Highly correlated variables to exclude from reliability analysis
    # correlations(data)
    data = clean(data, settings)
    
    # Weekend flag
    data['weekend'] = data.apply(lambda row: add_weekend_flag(row['day']), axis=1)

    for week_type, value  in {'Weekday': [0, 5], 'Weekend': [1,2], 'All': [None, 6]}.items():
        for group in data['group'].unique():
            tmp_data = data.loc[data['group'] == group]
            if value[0] is not None:
                tmp_data = tmp_data.loc[tmp_data['weekend'] == value[0]] # Only include specified weekend type for reliability analysis
            print(f'Group: {group}')
            ICC_analysis(tmp_data, group, week_type, value[1], settings)
            # weekday_weekend_ttest(tmp_data, group, settings)


#%%
