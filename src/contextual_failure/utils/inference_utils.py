import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.sampling import sample_hypersphere
from botorch.utils.transforms import t_batch_mode_transform
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from botorch.models import SingleTaskVariationalGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RBFKernel, ScaleKernel
from torch.quasirandom import SobolEngine
from botorch.models.transforms.outcome import Standardize
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import os
from scipy.spatial.distance import cdist
from torch.optim import Adam
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from gpytorch.likelihoods import MultitaskGaussianLikelihood

def smooth_mask(x, a, eps=2e-3):
    """Returns 0ish for x < a and 1ish for x > a"""
    return torch.nn.Sigmoid()((x - a) / eps)


def smooth_box_mask(x, a, b, eps=2e-3):
    """Returns 1ish for a < x < b and 0ish otherwise"""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)



class MutualInformation:
    def __init__(self,
                 model,
                 likelihood,
                 option = 'exact',
                 num_gp_samples: int = 128,
                 num_likelihood_samples: int = 128,
                 ):
        self.model = model
        self.likelihood = likelihood
        self.option = option
        self.num_gp_samples = num_gp_samples
        self.num_likelihood_samples = num_likelihood_samples

    def forward(self, X: torch.Tensor):
        """
        Compute the MI for input X of shape [b, q, d].

        Args:
        X -- input tensor of shape [b, q, d], where
            b: batch size,
            q: number of query points per batch,
            d: input dimensionality
        """
        if self.option == 'variational':
            f_samples = self.model.get_batched_samples(X, num_samples=torch.Size([self.num_gp_samples]))
        else:
            # Compute the posterior over f(x)
            posterior = self.model.posterior(X.float())
            # Sample f(x) from the posterior
            f_samples = posterior.rsample(
                sample_shape=torch.Size([self.num_gp_samples]))  # Shape: (num_gp_samples, b, q, o)

        # Compute H(y(x)) - Marginal entropy
        y_marginal_dist = self.likelihood(f_samples)  # Normal, Shape: (num_gp_samples, b, q, o)
        y_marginal_samples = y_marginal_dist.sample(
            sample_shape=torch.Size([self.num_likelihood_samples]))  # Shape: (num_likelihood_samples, num_gp_samples, b, q, o)
        y_marginal_log_probs = y_marginal_dist.log_prob(y_marginal_samples)  # Shape: (num_likelihood_samples, num_gp_samples, b, q)
        H_y = -torch.mean(torch.mean(y_marginal_log_probs, dim=1), dim=0)  # First marginalize f, then marginalize y
        H_y = torch.mean(H_y, dim=-1)  # Shape: (b)

        # Compute H(y(x) | f(x)) - Conditional entropy
        y_conditional_dists = self.likelihood(f_samples)  # Normal, Shape: (num_gp_samples, b, q, o)
        y_conditional_samples = y_conditional_dists.sample(
            sample_shape=torch.Size([self.num_likelihood_samples]))  # Shape: (num_likelihood_samples, num_gp_samples, b, q, o)
        y_conditional_log_probs = y_conditional_dists.log_prob(
            y_conditional_samples)  # Shape: (num_likelihood_samples, num_gp_samples, b, q)
        H_y_given_f = -torch.mean(torch.mean(y_conditional_log_probs, dim=0), dim=0)  # First marginalize y, then marginalize f
        H_y_given_f = torch.mean(H_y_given_f, dim=-1)  # Shape: (b)

        # Mutual Information
        mi = H_y - H_y_given_f  # Shape: (b)

        return mi, f_samples


class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        constraints,
        punchout_radius,
        bounds,
        train_inputs,
        train_Y,
        lambda_=0.0,
        num_samples=128,
        **kwargs,
    ):
        """Expected Coverage Improvement (q=1 required, analytic)

        Right now, we assume that all the models in the ModelListGP have
        the same training inputs.

        Args:
            model: A ModelListGP object containing models matching the corresponding constraints.
                All models are assumed to have the same training data.
            constraints: List containing 2-tuples with (direction, value), e.g.,
                [('gt', 3), ('lt', 4)]. It is necessary that
                len(constraints) == model.num_outputs.
            punchout_radius: Positive value defining the desired minimum distance between points
            bounds: torch.tensor whose first row is the lower bounds and second row is the upper bounds
            num_samples: Number of samples for MC integration
        """
        super().__init__(model=model, objective=IdentityMCObjective(), **kwargs)
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        assert punchout_radius > 0
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.lambda_ = lambda_
        self.base_points = train_inputs
        self.base_points_y = train_Y
        self.likelihood = MultitaskGaussianLikelihood(model.num_outputs)
        self.mi = MutualInformation(model, self.likelihood)
        self.num_samples = num_samples
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        assert (
            all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        )

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def dim(self):
        return self.base_points.shape[-1]

    @property
    def train_inputs(self):
        return self.model.models[0].train_inputs[0]

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.double
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, **tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z
    
    def _generate_ball_of_points_metric(
        self, num_samples, radius, model,device=None, dtype=torch.double
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, model=model,**tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _get_base_point_mask(self, X):
        distance_matrix = self.model.models[0].covar_module.covar_dist(
            X.float(), self.base_points.float()
        )
        # distance_matrix2 = self.model.models[1].covar_module.covar_dist(
        #     X.float(), self.base_points.float()
        # )
        # distance_matrix = torch.stack((distance_matrix1,distance_matrix2),dim=-1)
        # distance_matrix = cdist(X.squeeze().cpu().float(),self.base_points.cpu().float())
        # breakpoint()
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _get_base_point_mask_metric(self,X):
        posterior_X = self.model.posterior(X.float()).mean
        # posterior_base = self.model.posterior(self.base_points.float()).mean
        posterior_base = self.base_points_y.float()
        distance_matrix = cdist(posterior_X.detach().cpu().reshape(-1,posterior_X.shape[-1]),posterior_base.detach().cpu())
        return smooth_mask(torch.tensor(distance_matrix), self.punchout_radius).reshape(X.shape[0],X.shape[1],-1)
        
        # cdist(self.base_points.)
                                    
    # def _get_base_point_mask(self, X):
    #     cost_X = self.model.models[0].posterior(
    #         X.float())
    #     cost_base_points = self.model.models[0].posterior(
    #         self.base_points.float())
    #     cdist_ = cdist(cost_X.cpu().detach().numpy(), cost_base_points.cpu().detach().numpy())
        
    #     breakpoint()
    #     return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points.float())
        mus, sigma2s = posterior.mean, posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )
        return probs

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate Expected Improvement on the candidate set X."""
        # lambda_ = 0.2
        beta=20
        alpha=1
        mi, y_samples = self.mi.forward(X)
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_box_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        base_point_mask2 = self._get_base_point_mask_metric(ball_around_X).prod(dim=-1)#.to(device='cuda')
        base_point_mask1 = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        # masked_prob = prob*domain_mask*base_point_mask2
        # masked_prob = prob*domain_mask*base_point_mask2
        masked_prob = prob * domain_mask * ((1-self.lambda_)*base_point_mask2+ self.lambda_*base_point_mask1)
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        efe = beta * mi + alpha*y.squeeze(-1)
        return efe



def get_and_fit_vgp(X, Y):
    """Creates and fits a Variational Gaussian Process (VGP) with one output dimension.

    X is assumed to be in [0, 1]^d, and Y should have shape (n, 1).
    """
    y_mean, y_std = Y.mean(), Y.std()
    Y = (Y - y_mean) / y_std  # Standardize
    # Y = Y.unsqueeze(-1)
    print("Y",Y.shape)

    # Define the GP model
    likelihood = GaussianLikelihood()
    octf = Standardize(m=1)
    model = SingleTaskVariationalGP(train_X=X, train_Y=Y, likelihood=likelihood, outcome_transform=octf,inducing_points=X[0:5,:])
    model.covar_module = RBFKernel(ard_num_dims=2,lengthscale=torch.tensor(0.1))

    # Define the marginal log likelihood
    mll = VariationalELBO(likelihood, model.model, num_data=Y.size(0))

    # Train the model
    fit_gpytorch_mll(mll)
    # Training Loop
    # model.train()
    # likelihood.train()
    # optimizer = Adam(model.parameters(), lr=0.1)

    # num_epochs = 1000
    # for i in range(num_epochs):
    #     optimizer.zero_grad()
    #     output = model(X)
    #     loss = -mll(output, Y.T)
    #     loss.backward()
    #     optimizer.step()

    return model

def get_and_fit_gp(X, Y):
    """Simple method for creating a GP with one output dimension.

    X is assumed to be in [0, 1]^d.
    """
    # assert Y.ndim == 2 and Y.shape[-1] == 1
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))  # Noise-free
    octf = Standardize(m=1)
    gp = SingleTaskGP(X, Y, likelihood=likelihood,outcome_transform=octf)
    mll = ExactMarginalLogLikelihood(model=gp, likelihood=gp.likelihood)
    fit_gpytorch_mll(mll)
    return gp

def yf(x):
    return (1 - torch.exp(-4 * (x[:, 0] - 0.4) ** 2)).unsqueeze(-1)

def identify_samples_which_satisfy_constraints(X, constraints):
    """
    Takes in values (a1, ..., ak, o) and returns (a1, ..., ak, o)
    True/False values, where o is the number of outputs.
    """
    successful = torch.ones(X.shape).to(X)
    for model_index in range(X.shape[-1]):
        these_X = X[..., model_index]
        direction, value = constraints[model_index]
        successful[..., model_index] = (
            these_X < value if direction == "lt" else these_X > value
        )
    return successful

if __name__ == "__main__":
    tkwargs = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.double,
    }
    x = torch.tensor([0, 0.15, 0.25, 0.4, 0.8, 1.0], **tkwargs).unsqueeze(-1)
    y = yf(x)
    gp = get_and_fit_gp(x, y)
    print("Working with 1D data")
    def yf2d(x):
        v1 = torch.exp(-2 * (x[:, 0] - 0.3) ** 2 - 4 * (x[:, 1] - 0.6) ** 2)
        v2 = torch.exp(-2 * (x[:, 0] - 0.6) ** 2 - 4 * (x[:, 1] - 0.3) ** 2)
        return torch.stack((v1, v2), dim=-1)


    bounds = torch.tensor([[0, 0], [1, 1]], **tkwargs)
    lb, ub = bounds
    dim = len(lb)
    constraints = [("lt", 0.4), ("gt", 0.55)]
    punchout_radius = 0.05

    SMOKE_TEST = os.environ.get("SMOKE_TEST")
    num_init_points = 5
    num_total_points = 30 if not SMOKE_TEST else 5

    X = lb + (ub - lb) * SobolEngine(dim, scramble=True).draw(num_init_points).to(**tkwargs)
    Y = yf2d(X)
    breakpoint()
    print(X.shape,Y.shape)

    while len(X) < num_total_points:
        # We don't have to normalize X since the domain is [0, 1]^2. Make sure to
        # appropriately adjust the punchout radius if the domain is normalized.
        gp_models = [get_and_fit_gp(X, Y[:, i : i + 1].reshape(-1,1)) for i in range(Y.shape[-1])]
        model_list_gp = ModelListGP(gp_models[0],gp_models[1])
        eci = ExpectedCoverageImprovement(
            model=model_list_gp,
            constraints=constraints,
            punchout_radius=punchout_radius,
            bounds=bounds,
            train_inputs=X,
            num_samples=128 if not SMOKE_TEST else 4,
        )
        x_next, _ = optimize_acqf(
            acq_function=eci,
            bounds=bounds,
            q=1,
            num_restarts=10 if not SMOKE_TEST else 2,
            raw_samples=512 if not SMOKE_TEST else 4,
        )
        y_next = yf2d(x_next)
        X = torch.cat((X, x_next))
        Y = torch.cat((Y, y_next))
    import matplotlib.pyplot as plt
    plt.plot(Y[:,0].cpu(), Y[:,1].cpu(), "o")
    plt.xlabel("$f_1(x)$")
    plt.ylabel("$f_2(x)$")
    plt.title("Sampled points")
    plt.savefig('test_param.png')
    # plt.show()
    
    N1, N2 = 30, 30
    Xplt, Yplt = torch.meshgrid(
        torch.linspace(0, 1, N1, **tkwargs), torch.linspace(0, 1, N2, **tkwargs)
    )
    xplt = torch.stack(
        (
            torch.reshape(Xplt, (Xplt.shape[0] * Xplt.shape[1],)),
            torch.reshape(Yplt, (Yplt.shape[0] * Yplt.shape[1],)),
        ),
        dim=1,
    )
    yplt = yf2d(xplt)
    Zplt = torch.reshape(yplt[:, 0], (N1, N2))  # Since f1(x) = f2(x)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    h1 = ax.contourf(Xplt.cpu().numpy(), Yplt.cpu().numpy(), Zplt.cpu().numpy(), 20, cmap="Blues", alpha=0.6)
    fig.colorbar(h1)
    ax.contour(Xplt.cpu().numpy(), Yplt.cpu().numpy(), Zplt.cpu().numpy(), [0.55, 0.75], colors="k")

    feasible_inds = (
        identify_samples_which_satisfy_constraints(Y, constraints)
        .prod(dim=-1)
        .to(torch.bool)
    )
    ax.plot(X[feasible_inds, 0].cpu(), X[feasible_inds, 1].cpu(), "sg", label="Feasible")
    ax.plot(
        X[~feasible_inds, 0].cpu(), X[~feasible_inds, 1].cpu(), "sr", label="Infeasible"
    )

    ax.legend(loc=[0.7, 0.05])
    ax.set_title("$f_1(x)$")  # Recall that f1(x) = f2(x)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", "box")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    # plt.show()
    plt.savefig('test_param_2.png')
    print("Number of points: ", len(X))
    print("Number of points in the feasible region: ", feasible_inds.sum())
    print("Number of points in the infeasible region: ", (~feasible_inds).sum())
    print("Number of points in the feasible region: ", feasible_inds.sum())