# Copyright (c) 2017 Wu Jun (littlewj187@gmail.com)

import numpy as np


class LCS:
    def __init__(self, window=20, beta=0.8, svd_fraction=0.99):
        """
        It initiates the local correlation tracking algorithm by the parameters below.
        :param window: the window size of the auto-correlation matrix.
        :param beta: the exponential smoothing coefficient.
        :param svd_fraction: the threshold of the SVD.
        """
        self.window = window
        self.beta = beta
        self.svd_fraction = svd_fraction
        self.instance_cnt = 0
        self.scores = []
        self.series1 = []
        self.series2 = []
        pass
    def lcs(self):
        """
        It implements local correlation scores and measure each point one by one.
        :return:
        """
        X = np.array(self.series1)
        Y = np.array(self.series2)

        last_matrix_X= np.array([[0] * self.window] * self.window)
        last_matrix_Y = np.array([[0] * self.window] * self.window)
        
        for i in range(self.window - 1, self.instance_cnt):
            #step1 - compute the current autocorrelation matrix.
            auto_matrix_X = self.compute_auto_matrix(X, i, last_matrix_X)
            auto_matrix_Y = self.compute_auto_matrix(Y, i, last_matrix_Y)
            #setp2: svd decomposition
            UX, SX, VX = np.linalg.svd(auto_matrix_X)
            UY, SY, VY = np.linalg.svd(auto_matrix_Y)
            SX = SX.T
            SY = SY.T
            SX = np.diag(SX)
            SY = np.diag(SY)
            SXDiagSum = 0.0
            SYDiagSum = 0.0
            for j in range(len(SX)): 
                SXDiagSum = SXDiagSum + SX[j][j]
            for j in range(len(SY)): 
                SYDiagSum = SYDiagSum + SY[j][j]
            SX = SX / SXDiagSum
            SY = SY / SYDiagSum
            #step3: calculate k
            SX_k = 1
            SY_k = 1
            fsum = 0.0
            for j in range(len(SX)):
                fsum = fsum + SX[j][j]
                if fsum >= self.svd_fraction:
                    SX_k = j + 1
                    break
            fsum = 0.0
            for j in range(0,len(SY)): 
                fsum = fsum + SY[j][j]
                if fsum >= self.svd_fraction:
                    SY_k = j + 1
                    break
            # setp4: compute local correlation score
            UXk = UX[0:len(UX),0:SX_k]
            UYk = UY[0:len(UY),0:SY_k]
            Ux1 = UX[0:len(UX),0:1]
            Uy1 = UY[0:len(UY),0:1]
            
            UXdotUy = np.dot(UXk.T,Uy1)
            UYdotUx = np.dot(UYk.T,Ux1)
            UxdotUx = np.dot(Ux1.T,Ux1)
            UydotUy = np.dot(Uy1.T,Uy1)
            
            self.scores[i] = 0.0
            norm_UXdotUy = 0.0
            norm_UYdotUx = 0.0
            norm_UxdotUx = UxdotUx[0][0]
            norm_UydotUy = UydotUy[0][0]  
            for j in range(len(UXdotUy)):
                norm_UXdotUy = norm_UXdotUy + UXdotUy[j][0] * UXdotUy[j][0]
            for j in range(len(UYdotUx)):
                norm_UYdotUx = norm_UYdotUx + UYdotUx[j][0] * UYdotUx[j][0]
            norm_UXdotUy = np.sqrt(norm_UXdotUy)
            norm_UYdotUx = np.sqrt(norm_UYdotUx)
            norm_UxdotUx = np.sqrt(norm_UxdotUx)
            norm_UydotUy = np.sqrt(norm_UydotUy)
            score1 = norm_UXdotUy / norm_UydotUy
            score2 = norm_UYdotUx / norm_UxdotUx
            self.scores[i] = 1.0 - 0.5 * (score1 + score2)
            last_matrix_X = auto_matrix_X
            last_matrix_Y = auto_matrix_Y

    def compute_auto_matrix(self, X, t, last_matrix):
        """
        It computes the auto-correlation matrix for X.
        :param X: list<float> | a time series.
        :param t: int | the index of the current point.
        :param last_matrix: array<float> | the last auto-correlation matrix.
        :return: array<float> | the the auto-correlation
        """
        win = self.window
        beta = self.beta
        lft_idx = t - win + 1
        if lft_idx < 0 or lft_idx + win > len(X):
            return last_matrix
        mat = []
        for l1 in range(0, win):
            line = []
            for l2 in range(0, win):
                line.append(X[l1 + lft_idx] * X[l2 + lft_idx])
            mat.append(line)
        ans_matrix = last_matrix * beta + np.array(mat)
        return ans_matrix

    def measure_current_period(self, series1, series2, labels, period_len):
        """
        It computes the local correlation score of the current period(iteration).
        :param series1: list<float> | the time series 1.
        :param series2: list<float> | the time series 2.
        :param labels: list<bool> | the labels of the previous instances.
        :param period_len: int | the duration of a period.
        :return: list<float> | the local correlation score of the current period.
        """
        _series1 = []
        _series2 = []
        for i in range(len(labels)):
            if not labels[i]:
                _series1.append(series1[i])
                _series2.append(series2[i])
        for i in range(len(labels), len(series1)):
            _series1.append(series1[i])
            _series2.append(series2[i])
        self.instance_cnt = len(_series1)
        self.scores = [0.0] * self.instance_cnt
        self.series1 = _series1
        self.series2 = _series2
        self.lcs()
        return self.scores[-period_len:]
