import torch
from sklearn.ensemble import RandomForestClassifier
from mixed_detection.utils import getClassificationMetrics, process_output, update_regression_features
from mixed_detection.calibration import logregCal
import numpy as np
from tqdm import tqdm

EPSILON = 1e-100


class BinaryClassifier():

    def __init__(self,expected_prevalence, costs_ratio):
        self.x = None
        self.y = None
        self.clf = RandomForestClassifier(random_state=32)
        self.used_features = 7  #Between 1 and 7
        self.x_binary_cont = None
        self.x_positive_posteriors = None
        self.calibration_parameters = {}

        self.expected_prevalence = expected_prevalence
        self.costs_ratio = costs_ratio
        tau_bayes = self.costs_ratio * (1 - self.expected_prevalence) / self.expected_prevalence
        self.posteriors_th = tau_bayes / (1 + tau_bayes)

        self.train_positive_prior = 0.5
        self.train_negative_prior = 0.5



    def train(self):
        self.fit_classifier()
        self.calibrate()

    def infere(self,x_regresion_test):
        assert len(self.calibration_parameters) > 0, "Classifier was not trained yet"

        x_binary_cont = self.clf.predict_proba(x_regresion_test[:, :self.used_features])
        positive_posteriors = x_binary_cont[:, 1]
        negative_posteriors = x_binary_cont[:, 0]

        LLR = np.log((positive_posteriors + EPSILON) / (negative_posteriors + EPSILON)) - np.log(
            (self.train_positive_prior + EPSILON) / (self.train_negative_prior + EPSILON))
        a = self.calibration_parameters['a']
        b = self.calibration_parameters['b']
        k = self.calibration_parameters['k']

        x_positive_posteriors = 1 / (1 + np.exp(-(a * LLR + b) + k))
        x_binary = [1 if x > self.posteriors_th else 0 for x in x_positive_posteriors]
        return x_binary, x_positive_posteriors

    def set_data(self,x,y):
        self.x = x
        self.y = y
        self.train_positive_prior = len(np.argwhere(self.y == 1)) / len(self.y)
        self.train_negative_prior = len(np.argwhere(self.y == 0)) / len(self.y)
        assert self.train_negative_prior + self.train_positive_prior == 1, "Error calculating train priors, len tar {} len non ".format(len(np.argwhere(self.y==1)),
                                                                                                                              len(np.argwhere(self.y==0)))

    def get_data_from_model(self,model,data_loader,device):
        x_regresion = np.zeros((len(data_loader.dataset), self.used_features))
        y_regresion = np.zeros(len(data_loader.dataset))
        cpu_device = torch.device("cpu")
        model.eval()
        j = 0
        image_paths =[]
        with torch.no_grad():
            for batch in tqdm(data_loader):
                if data_loader.dataset.return_image_source:
                    images, targets, image_sources, batch_paths = batch
                    if isinstance(batch_paths, tuple):
                        image_paths += list(batch_paths)
                    if isinstance(batch_paths, list):
                        image_paths += batch_paths
                    if isinstance(batch_paths, str):
                        image_paths.append(batch_paths)
                else:
                    images, targets = batch
                images = list(img.to(device) for img in images)
                torch.cuda.synchronize()
                outputs = model(images)
                outputs = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in outputs]
                targets = [{k: v.to(cpu_device).detach() for k, v in t.items()} for t in targets]

                for img_id, output in enumerate(outputs):
                    height = images[img_id].shape[1]
                    width = images[img_id].shape[2]
                    total_area = height * width
                    output = process_output(output, total_area,
                                             max_detections=None,
                                             min_box_proportionArea=None,
                                             min_score_threshold=None
                                             )
                    # print('beofre target',psutil.virtual_memory().percent)
                    target = targets[img_id]
                    N_targets = len(target['boxes'].detach().numpy())
                    gt = 1 if N_targets > 0 else 0
                    y_regresion[j] = gt

                    # print('before scores',psutil.virtual_memory().percent)
                    image_scores = output['scores']  # .detach().numpy()
                    image_areas = output['areas']  # .detach().numpy()
                    x_regresion[j, :] = update_regression_features(image_scores, image_areas,n_features=self.used_features)
                    j += 1
                    del gt, image_scores, image_areas, target
                del images, targets, outputs
        self.x = x_regresion
        self.y = y_regresion
        self.train_positive_prior = len(np.argwhere(self.y == 1)) / len(self.y)
        self.train_negative_prior = len(np.argwhere(self.y == 0)) / len(self.y)
        assert self.train_negative_prior + self.train_positive_prior == 1, "Error calculating train priors, len tar {} len non ".format(len(np.argwhere(self.y==1)),
                                                                                                                              len(np.argwhere(self.y==0)))

    def reset_params(self,expected_prevalence,costs_ratio):
        self.expected_prevalence = expected_prevalence
        self.costs_ratio = costs_ratio
        self.calibrate()

    def fit_classifier(self):
        assert self.x is not None and self.y is not None
        self.clf.fit(self.x[:, :self.used_features ], self.y)
        self.x_binary_cont = self.clf.predict_proba(self.x[:, :self.used_features])

    def calibrate(self):
        assert self.x_binary_cont is not None
        positive_posteriors = self.x_binary_cont[:, 1]
        negative_posteriors = self.x_binary_cont[:, 0]

        LLR = np.log((positive_posteriors + EPSILON) / (negative_posteriors + EPSILON)) - np.log(
            (self.train_positive_prior + EPSILON) / (self.train_negative_prior + EPSILON))

        tar = LLR[self.y == 1]
        non = LLR[self.y == 0]
        print('Len tar {} Len non {}'.format(len(tar), len(non)))
        theta = np.log(self.costs_ratio * (1 - self.expected_prevalence) / self.expected_prevalence)
        ptar_hat = 1 / (1 + np.exp(theta))

        # Fit a linear calibrator to the set
        a, b = logregCal(tar, non, ptar_hat, return_params=True)
        k = -np.log((1 - self.expected_prevalence) / self.expected_prevalence)
        print('a {:.2f} b {:.2f} k {:.2f}'.format(a, b, k))
        self.calibration_parameters = {'a': a, 'b': b, 'k': k}
        self.x_positive_posteriors = 1 / (1 + np.exp(-(a * LLR + b) + k))



