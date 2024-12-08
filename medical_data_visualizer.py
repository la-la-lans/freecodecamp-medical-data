import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
df['BMI'] =  df['weight'] / (df['height'] / 100)**2
df['overweight'] = (df['BMI'] > 25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars = ['cardio'], 
    value_vars =['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio','variable','value']).size().reset_index(name='total')

    # 7
    fig = sns.catplot(
    data=df_cat,
    x='variable', y='total',
    col='cardio',
    hue='value', kind='bar').fig

    # 8
    fig.savefig('catplot.png')
    return fig

# 9
df = df.drop(columns = ['BMI'])

def draw_heat_map():
    # 10
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])&
             (df['height'] >= df['height'].quantile(0.025)) & 
             (df['height'] <= df['height'].quantile(0.975)) & 
             (df['weight'] >= df['weight'].quantile(0.025)) & 
             (df['weight'] <= df['weight'].quantile(0.975))].reset_index(drop=True)
    # 11
    corr = df_heat.corr()
    # 12
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # 13
    fig, ax = plt.subplots()

    # 14
    sns.heatmap(corr, mask=mask, annot=True, linewidth=.3, ax=ax, fmt='.1f')

    # 15
    fig.savefig('heatmap.png')
    return fig
