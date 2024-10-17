import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_dataframe(df, filename):
    df.to_pickle(filename)
    print(f"DataFrame saved to {filename}")

def load_dataframe(filename):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        return df
    else:
        print(f"File {filename} not found.")
        return None

def fetch_wandb_data(project_name, tags, metrics, agent_param_count_metric):
    api = wandb.Api(timeout=60)
    runs = api.runs(project_name, {"tags": {"$in": tags}})
    print(f"found {len(runs)} runs with tags {tags}")
    
    all_run_data = []
    for run in runs:
        print(run.name)
        print(run.config)

        # QMIX/DAgger
        hyperaware = run.config.get("alg")["AGENT_HYPERAWARE"]
        cap_aware = run.config.get("env")["ENV_KWARGS"]["capability_aware"]
        init_scale = run.config.get("alg")["AGENT_HYPERNET_KWARGS"]["INIT_SCALE"]

        # MAPPO
        # hyperaware = run.config.get("AGENT_HYPERAWARE")
        # cap_aware = run.config.get("ENV_KWARGS")["capability_aware"]
        # init_scale = run.config.get("AGENT_HYPERNET_KWARGS")["INIT_SCALE"]

        # drop the one CASH run with init_scale = 0.5!
        if (hyperaware and cap_aware) and init_scale != 0.2:
            print(f"\nDropping {run.name} w/ init_scale != 0.2!\n")
            continue

        # NOTE: agent_param_count is only plotted once and thus cannot be found in the same scan_history() call as the other metrics, finding manually here:
        agent_param_count = run.scan_history(keys=[agent_param_count_metric])
        agent_param_count = next(agent_param_count)[agent_param_count_metric]
        print('agent_param_count', agent_param_count)

        # get metrics of run over time
        history = run.scan_history(keys=metrics)

        run_data = pd.DataFrame(history)
        run_data['run_id'] = run.id
        run_data['run_name'] = run.name
        run_data['tags'] = ', '.join(run.tags)
        run_data['hyperaware'] = hyperaware
        run_data['cap_aware'] = cap_aware
        run_data['agent_param_count'] = agent_param_count

        # for DAgger I didn't log TS, add that (hardcoded based on manual math...)
        run_data['timestep'] = 10 * run_data['policy/updates']

        print(run_data.head())

        all_run_data.append(run_data)
    
    return pd.concat(all_run_data, ignore_index=True)

def baseline_name(row):
    if row['hyperaware']: 
        if row['cap_aware']:
            return "CASH"
    else:
        if row['cap_aware']:
            return "RNN-CA"
        else:
            return "RNN-UN"

def get_from_wandb():
    # get runs from wandb
    # NOTE: I've been hardcoding the right things in...

    project_name = "JaxMARL"

    # tags = ['final-qmix-fire']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_fire_success_rate', 'test_snd', 'test_pct_fires_put_out']
    # agent_param_count_metric = 'agent_param_count'

    # tags = ['final-qmix-hmt']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_makespan', 'test_snd', 'test_quota_met']
    # agent_param_count_metric = 'agent_param_count'

    # tags = ['final-mappo-fire']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_fire_success_rate', 'test_snd', 'test_pct_fires_put_out']
    # agent_param_count_metric = 'actor_param_count'

    # tags = ['final-mappo-hmt']
    # fire_metrics = ['timestep', 'returns', 'test_returns', 'test_makespan', 'test_snd', 'test_quota_met']
    # agent_param_count_metric = 'actor_param_count'

    # tags = ['final-dagger-fire']
    # fire_metrics = ['policy/updates', 'policy/returns', 'policy/fire_success_rate', 'policy/snd', 'policy/pct_fires_put_out']
    # agent_param_count_metric = 'policy/agent_param_count'

    tags = ['final-dagger-hmt']
    dagger_hmt_metrics = ['policy/updates', 'policy/returns', 'policy/snd', 'policy/makespan', 'policy/quota_met', 'policy/loss']
    agent_param_count_metric = 'policy/agent_param_count'

    df = fetch_wandb_data(project_name, tags, dagger_hmt_metrics, agent_param_count_metric)
    filename = f"{tags[0]}.pkl"
    save_dataframe(df, filename)

    print("saved")
    print(df.head())

def smooth_and_downsample(df, y_column, mean_window=50, std_window=50, downsample_factor=10):
    """
    Creates a new dataframe with smoothed and downsampled data, with separate
    smoothing controls for mean and standard deviation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    y_column : str
        Column to analyze
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    """
    smoothed_data = []
    df_copy = df.copy()
    
    for baseline in df_copy['baseline'].unique():
        baseline_data = df_copy[df_copy['baseline'] == baseline].copy()
        baseline_data = baseline_data.sort_values('timestep')
        
        # Group by timestep to calculate mean and std
        grouped = baseline_data.groupby('timestep')[y_column].agg(['mean', 'std']).reset_index()
        
        # Smooth mean and std separately
        grouped['smooth_mean'] = grouped['mean'].rolling(
            window=mean_window, min_periods=1, center=True).mean()
        grouped['smooth_std'] = grouped['std'].rolling(
            window=std_window, min_periods=1, center=True).mean()
        
        # Downsample
        grouped = grouped.iloc[::downsample_factor]
        
        # Create dataframe with smoothed mean and smoothed std
        smoothed_df = pd.DataFrame({
            'timestep': grouped['timestep'],
            f'{y_column}': grouped['smooth_mean'],
            f'{y_column}_std': grouped['smooth_std'],
            'baseline': baseline
        })
        
        smoothed_data.append(smoothed_df)
    
    return pd.concat(smoothed_data)

def plot_metrics(df, y_label, y_column, title, mean_window, std_window, downsample_factor, alpha=0.2):
    """
    Generic plotting function for any metric with separate smoothing controls
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    y_column : str
        The name of the column to plot on y-axis
    title : str
        The title of the plot
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    alpha : float
        Transparency of the standard deviation bands (0-1)
    """
    # Create smoothed version of the data for plotting
    smoothed_df = smooth_and_downsample(df, y_column=y_column, 
                                      mean_window=mean_window,
                                      std_window=std_window,
                                      downsample_factor=downsample_factor)
    
    plt.figure(figsize=(6.4, 4.8))
    
    base_palette = sns.color_palette()
    palette = {
        'CASH': base_palette[2],
        'RNN-CA': base_palette[0],
        'RNN-UN': base_palette[1],
    }
    
    # Plot each baseline separately
    for baseline in smoothed_df['baseline'].unique():
        baseline_data = smoothed_df[smoothed_df['baseline'] == baseline]
        color = palette[baseline]
        
        # Plot mean line
        plt.plot(baseline_data['timestep'], baseline_data[y_column], 
                color=color, label=baseline, linewidth=2)
        
        # Add error bands with smoothed std
        plt.fill_between(baseline_data['timestep'],
                        baseline_data[y_column] - baseline_data[f'{y_column}_std'],
                        baseline_data[y_column] + baseline_data[f'{y_column}_std'],
                        color=color, alpha=alpha)
    
    plt.xlabel('Timestep')
    plt.ylabel(y_label)
    plt.title(title)
    # plt.legend(loc='best').set_draggable(True)
    plt.tight_layout(pad=0.5)

    # plt.show()
    plt.savefig(f'{title}-{y_label}.png'.lower().replace(' ', '-'))

def plot_from_saved():
    # load preprocessed data
    filename = "final-mappo-hmt.pkl"
    df = load_dataframe(filename)

    pd.set_option('display.max_columns', None)
    print(df.head())
    
    # translate baseline names
    df['baseline'] = df.apply(lambda row: f"{baseline_name(row)}", axis=1)

    # Fire
    # plot_metrics(df, y_label='Training Returns', y_column='returns', title='Firefighting', mean_window=100, std_window=100, downsample_factor=10)
    # plot_metrics(df, y_label='Test Returns', y_column='test_returns', title='Firefighting', mean_window=100, std_window=100, downsample_factor=10)
    # plot_metrics(df, y_label='SND', y_column='test_snd', title='Firefighting', mean_window=100, std_window=100, downsample_factor=10)
    # plot_metrics(df, y_label='Success Rate', y_column='test_fire_success_rate', title='Firefighting', mean_window=100, std_window=100, downsample_factor=10)
    # plot_metrics(df, y_label='Pct of Fires Extinguished', y_column='test_pct_fires_put_out', title='Firefighting', mean_window=100, std_window=100, downsample_factor=10)

    # HMT
    plot_metrics(df, y_label='Training Returns', y_column='returns', title='Transport', mean_window=100, std_window=100, downsample_factor=10)
    plot_metrics(df, y_label='Test Returns', y_column='test_returns', title='Transport', mean_window=100, std_window=100, downsample_factor=10)
    plot_metrics(df, y_label='SND', y_column='test_snd', title='Transport', mean_window=100, std_window=100, downsample_factor=10)
    plot_metrics(df, y_label='Success Rate', y_column='test_quota_met', title='Transport', mean_window=100, std_window=100, downsample_factor=10)
    plot_metrics(df, y_label='Makespan', y_column='test_makespan', title='Transport', mean_window=100, std_window=100, downsample_factor=10)


if __name__ == "__main__":
    # get_from_wandb()
    plot_from_saved()

