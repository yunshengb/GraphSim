from utils import load

fp = '/xxx/model/Siamese/logs/siamese_regression_xxx/test_info.pickle'

d = load(fp)
model = 'siamese_regression'
r = d['test_results']
s = d['sim_mat']
print(r['mse_norm'][model])
print(r['preck_norm_0'][model]['precs'][9])
print(r['preck_norm_0'][model]['precs'][19])
