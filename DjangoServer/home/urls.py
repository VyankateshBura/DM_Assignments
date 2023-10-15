
from django.urls import path,include
from . import views

urlpatterns = [
    path('chisquare/',views.calculate_contingency_table,name = "contingency_view"),
    path('correlation/',views.correlation_analysis,name = "correlation_analysis"),
    path('zscore/',views.zscoreCalc,name = "zscoreCalc"),
    path('minmax/',views.minMaxNormalization,name = "minMaxNormalization"),
    path('decision-tree/',views.decision_tree_classifier,name = "decision_tree_classifier"),
    path('rulebased/',views.Rule_based_classifier,name = "Rule_based_classifier"),
    path('ann/',views.neural_network_classifier,name = "neural_network_classifier"),
    path('regression/',views.regression_classifier,name = "regression_classifier"),
    path('naivebayes/',views.naive_bayesian_classifier,name = "naive_bayesian_classifier"),
    path('knn/',views.knn_classifier,name = "knn_classifier"),
    path('decimalscaling/',views.decimalScalingNormalization,name = "decimalScalingNormalization"),
]