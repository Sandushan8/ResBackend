import numpy as np
from predictor import load_model, make_predictions
from hrvanalysis import remove_outliers, get_time_domain_features, get_frequency_domain_features,get_poincare_plot_features

def intiate_prediction(data):

    cleaned_rr_intervals = remove_outliers(data)
    time_domain_results = get_time_domain_features(cleaned_rr_intervals)
    frequency_domain_results = get_frequency_domain_features(cleaned_rr_intervals)
    poincare = get_poincare_plot_features(cleaned_rr_intervals)
    model = load_model('random_forest_model.pkl')
    sample_data = np.array([[
    time_domain_results['mean_nni'], time_domain_results['median_nni'],time_domain_results['rmssd'], time_domain_results['sdsd'],
    time_domain_results['pnni_20'], time_domain_results['pnni_50'], poincare['sd1'],
    poincare['sd2'], frequency_domain_results['vlf'], frequency_domain_results['lf'],
    frequency_domain_results['lfnu'], frequency_domain_results['hf'], frequency_domain_results['hfnu'],
    frequency_domain_results['total_power'], frequency_domain_results['lf_hf_ratio']
    ]]) 
    predictions = make_predictions(model, sample_data)
    print('pred', predictions[0])
    return  predictions[0]
