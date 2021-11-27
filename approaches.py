"""
    The methods to find the important features
"""

TARGET_THRESHOLD = 0.4

def correlated_features(data):
    data_cor = data.corr()
    cor_target = abs(data_cor['class'])
    high_corr_features = cor_target[cor_target > TARGET_THRESHOLD]
    feature_names = [index for index, _ in high_corr_features.items()]
    feature_names.remove('class')
    return feature_names

def select_k_best():
    pass