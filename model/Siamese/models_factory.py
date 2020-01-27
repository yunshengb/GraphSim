from model_regression import SiameseRegressionModel


def create_model(model, input_dim, data, dist_sim_calculator):
    if model in ['siamese_regression']:
        return SiameseRegressionModel(input_dim, data, dist_sim_calculator)
    else:
        raise RuntimeError('Unknown model {}'.format(model))
