from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.eotb_path = 'yourpath'
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = 'yourpath'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = 'yourpath'
    settings.results_path = 'yourpath'    # Where to store tracking results
    settings.segmentation_path = 'yourpath'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

