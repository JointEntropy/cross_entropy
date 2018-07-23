import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
from utils import inverse_ohe
from keras.callbacks import *
import shutil

# class LeaveOutAuthorsScore(Callback):
#     def __init__(self,
#                  train_data, test_data,
#                  clf,
#                  ohe,
#                  on_each=3):
#         self.clf = clf
#         self.train = train_data
#         self.test = test_data
#         self.ohe = ohe
#         self.on_each = on_each
#
#     def on_epoch_end(self, epoch, logs=None):
#
#         # Загружаем моделкьу
#         intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.layers[-1].input)
#         # Генерим признаки
#         print('Fitting on epoch {}'.format(epoch))
#         features_train = intermediate_layer_model.predict(self.train[0], verbose=1)
#         features_test = intermediate_layer_model.predict(self.test[0], verbose=1)
#         features_extratrain, features_extraval, test_l1, test_l2, = train_test_split(features_test, self.test[1],
#                                                                                      stratify=self.test[1])
#         combo_train_X = np.concatenate([features_train, features_extratrain], axis=0)
#         combo_train_y = np.concatenate([self.train[1], test_l1], axis=0)
#         # перемешиваем
#         tmp_idx = np.arange(combo_train_X.shape[0])
#         np.random.shuffle(tmp_idx)
#         combo_train_X, combo_train_y = combo_train_X[tmp_idx], combo_train_y[tmp_idx]
#
#         self.clf.fit(combo_train_X, inverse_ohe(combo_train_y, self.ohe))
#         print("Средняя точность на исключительно новых данных {}".format(
#             self.clf.score(features_extraval, inverse_ohe(test_l2, self.ohe))))
#


class AggregatedPredictions(Callback):
    def __init__(self, val_data, val_target, ohe,
                 on_each=3, best=5,
                 random=True,
                 verbose=1,
                 tmp_dir='tmp',
                 scorer=None):
        self.val_data = val_data  # preprocessed validation data with no index, but shuffled as comp_groups val
        self.val_target = val_target
        self.ohe = ohe
        self.scorer = scorer or accuracy_score
        self.on_each = on_each
        self.preds = []
        self.scores = []
        self.best = best
        self.best_ensemble_score = 0
        self.best_ensemble_epochs = []
        self.epochs = []
        self.random = random
        self.verbose = verbose


        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
        self.tmp_dir = tmp_dir

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.on_each != 0 or epoch == 0:
            return

        self.epochs.append(epoch)
        self.model.save(os.path.join(self.tmp_dir, 'tmp_{}.h5'.format(epoch)))

        pred = self.model.predict(self.val_data)
        self.preds.append(pred)

        pred = inverse_ohe(pred, self.ohe)
        score = self.scorer(self.val_target, pred)
        self.scores.append(score)
        if self.verbose >=2:
            print('Val score : {:.4f}'.format(score))


        # Выделяем self.best лучших предсказаний по локальному скору
        best_preds = sorted(list(zip(self.scores, self.preds, self.epochs)), key=lambda x: x[0])[-self.best:]
        mean_preds = np.array([pred for score, pred, epoch in best_preds]).mean(axis=0)
        pred = inverse_ohe(mean_preds, self.ohe)
        score = self.scorer(self.val_target, pred)
        if self.verbose >= 2:
            print('Val average score: {:.4f}'.format(score))

        if score > self.best_ensemble_score:
            self.best_ensemble_epochs = [epoch for score, pred, epoch in best_preds]
            self.best_ensemble_score = score
            if self.verbose >= 1 and self.best % epoch != 0:
                print('Best ensemble of {} has score {:.4f}'.format(self.best_ensemble_epochs, self.best_ensemble_score))

    def on_train_end(self, logs=None):
        for epoch in set(self.epochs) - set(self.best_ensemble_epochs):
            os.remove(os.path.join(self.tmp_dir,'tmp_{}.h5'.format(epoch)))
        print('Remain models from epochs: {}, which gain score {:.3f}'.format(self.best_ensemble_epochs,
                                                                              self.best_ensemble_score))


class CyclicLR(Callback):
    """
    https://github.com/bckenstler/CLR
    """
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())