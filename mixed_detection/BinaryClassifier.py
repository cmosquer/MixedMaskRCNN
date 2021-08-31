import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve
from mixed_detection.utils import getClassificationMetrics, process_output, update_regression_features
from mixed_detection.calibration import logregCal
import numpy as np
from tqdm import tqdm

EPSILON = 1e-100

BETA = 1/3  #RATIO IMPORTANCIA RECALL / IMPORTANCIA PRECISION
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
        self.feature_idx = -1 #-1 for random forest, else a feature index in x_regression

        self.use_calibrated = True

    def train(self):
        self.fit_classifier()
        self.calibrate()

    def infere(self,x_regresion_test):

        if self.feature_idx == -1:
            x_binary_cont = self.clf.predict_proba(x_regresion_test[:, :self.used_features])
            positive_posteriors = x_binary_cont[:, 1]
            negative_posteriors = x_binary_cont[:, 0]
        else:
            x_binary_cont = x_regresion_test[:,self.feature_idx]
            positive_posteriors = x_binary_cont
            negative_posteriors = 1-x_binary_cont
        if self.use_calibrated:
            assert len(self.calibration_parameters) > 0, "Classifier was not trained yet"
            LLR = np.log((positive_posteriors + EPSILON) / (negative_posteriors + EPSILON)) - np.log(
                (self.train_positive_prior + EPSILON) / (self.train_negative_prior + EPSILON))
            a = self.calibration_parameters['a']
            b = self.calibration_parameters['b']
            k = self.calibration_parameters['k']

            x_positive_posteriors = 1 / (1 + np.exp(-(a * LLR + b) + k))
        else:
            x_positive_posteriors = x_binary_cont.copy()
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
        if self.feature_idx == -1:
            positive_posteriors = self.x_binary_cont[:, 1]
            negative_posteriors = self.x_binary_cont[:, 0]
        else:
            positive_posteriors = self.x_binary_cont
            negative_posteriors = 1-self.x_binary_cont

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
        tau_bayes = self.costs_ratio * (1 - self.expected_prevalence) / self.expected_prevalence
        self.posteriors_th = tau_bayes / (1 + tau_bayes)

    def use_one_feature(self,feature_idx,threshold_method='calibrate'):
        """threshold_method: "calibrate" or "roc"
        """
        self.x_binary_cont = self.x[:,feature_idx]
        self.feature_idx = feature_idx
        if threshold_method=='calibrate':
            self.use_calibrated=True
            self.calibrate()
        if threshold_method == 'roc':
            fpr,tpr,th = roc_curve(self.y,self.x_binary_cont)
            self.posteriors_th = th[np.argmax(tpr-fpr)]
            self.use_calibrated = False
        if threshold_method == 'precision':
            precision,recall,th = precision_recall_curve(self.y,self.x_binary_cont)
            self.posteriors_th = th[np.argmax(precision)]
            self.use_calibrated = False
        if threshold_method == 'f1':
            precision,recall,th = precision_recall_curve(self.y,self.x_binary_cont)
            f1 = (1+BETA**2)*precision*recall/((BETA**2)*precision+recall)
            self.posteriors_th = th[np.argmax(f1)]
            self.use_calibrated = False
    def copy_data(self,binary_classifier):
        self.x = binary_classifier.x
        self.y = binary_classifier.y
        self.x_binary_cont = binary_classifier.x_binary_cont
        self.x_positive_posteriors = binary_classifier.x_positive_posteriors
        self.train_positive_prior = binary_classifier.train_positive_prior
        self.train_negative_prior = binary_classifier.train_negative_prior
