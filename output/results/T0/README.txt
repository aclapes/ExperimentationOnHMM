Check the effect of:

normParams = [
    0 0; 
    1 0; 
    2 0.1; 2 1; 2 10; 
    3 1; 3 2];
projVars = [0; 0.75; 0.9];
emInits = {'rnd';'kmeans'};
covTypes = {'diag';'full'};

Details:

- normalisation type (and some implicit parameter, e.g. scale on standardization or p-norm in unit scaling)
- decorrelation/dimensionality reduction with PCA
- Expectation-Maximization initialisation
- Covariance matrix type