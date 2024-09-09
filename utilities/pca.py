import warnings
import torch


class PCA:
    def __init__(self, n_components):
        self.pca_obj = SimplePCA(n_components)  # Can be replaced with more robust implementations, we didn't find it helpful
        self.explained_variance_ratio_ = None
        self.device_ = None
        self.trained_ = False
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Mean centering and variance normalization
        self.mean_ = torch.mean(X, dim=0)
        self.std_ = torch.std(X, dim=0) + 1e-9  # Adding a small value to prevent division by zero
        X -= self.mean_
        X /= self.std_ 

        try:
            self.pca_obj.fit(X)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA out of memory error while training PCA model. Trying to train on CPU...")
                X = X.cpu()
                self.pca_obj.fit(X)
            else:
                raise e

        self.device_ = X.device
        self.mean_, self.std_ = self.mean_.to(self.device_), self.std_.to(self.device_)
        self.explained_variance_ratio_ = self.pca_obj.explained_variance_ratio_
        self.trained_ = True
        return self
    
    def transform(self, X):
        if not self.trained_:
            raise RuntimeError("The model needs to be fitted before transforming!")
        
        input_device = X.device
        if X.device != self.device_:
            X = X.to(self.device_)
        
        X = (X - self.mean_) / self.std_
        return self.pca_obj.transform(X).to(input_device)
    

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



class SimplePCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        # Compute the SVD
        U, S, V = torch.svd(X)
        
        # Store the top `n_components` right singular vectors (principal directions)
        self.components_ = V.t()[:self.n_components]
        
        # Compute the explained variance ratio
        total_variance = torch.sum(S**2)
        explained_variance = (S**2)[:self.n_components]
        self.explained_variance_ratio_ = (explained_variance / total_variance).cpu().numpy()
        
        return self
    
    def transform(self, X):
        # Project data onto the principal directions
        return torch.mm(X, self.components_.t())

