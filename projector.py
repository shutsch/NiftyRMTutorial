import nifty8.re as jft
import jax.numpy as np


class PlaneProjector(jft.Model):
    def __init__(self, x, y, fov_shape, fov_resolution, reference_point):
        
        if not x.shape == y.shape:
            raise ValueError("x and y must have the same shape")
        
        self.reference_point = reference_point
        
        if reference_point is None:
            reference_point = np.array([0, 0])
            
        self.fov_shape = np.asarray(fov_shape)
        self.fov_resolution = np.asarray(fov_resolution)
            
        self.x_ind = np.asarray((x - reference_point[0])/self.fov_resolution[0]).astype(int)
        self.y_ind = np.asarray((y - reference_point[1])/self.fov_resolution[1]).astype(int)
            
        if np.any(self.x_ind < 0) or np.any(self.y_ind < 0):
            raise ValueError("x and y must be within the field of view, but some values are smaller than the reference point.")

        if np.any(self.x_ind >= fov_shape[0]) or np.any(self.y_ind >= fov_shape[1]):
            raise ValueError("x and y must be within the field of view, but some values are larger than the field of view boundaries.")

        super().__init__(domain=jft.ShapeWithDtype(fov_shape, np.float64), target=jft.ShapeWithDtype(self.x_ind.shape, np.float64))

    def __call__(self, x):
        return x.at[self.x_ind, self.y_ind].get() 
    
    def adjoint_times(self, x):
        # helper function to compute the adjoint of the call method
        xval = np.zeros(self.fov_shape, dtype=np.float64)
        return xval.at[self.x_ind, self.y_ind].add(x.val)
