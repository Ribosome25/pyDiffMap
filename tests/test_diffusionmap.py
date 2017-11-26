import numpy as np
import pytest

from pydiffmap import diffusion_map as dm
from scipy.sparse import csr_matrix


@pytest.fixture(scope='module')
def spherical_data():
    # Construct dataset
    phi = np.pi*np.linspace(-1, 1, 61)[1:]
    theta = np.pi*np.linspace(-1, 1, 33)[1:-1]
    Phi, Theta = np.meshgrid(phi, theta)
    Phi = Phi.ravel()
    Theta = Theta.ravel()

    X = np.cos(Theta)*np.cos(Phi)
    Y = np.cos(Theta)*np.sin(Phi)
    Z = np.sin(Theta)
    return np.array([X, Y, Z]).transpose(), Phi, Theta


@pytest.fixture(scope='module')
def uniform_2d_data():
    x = np.linspace(0., 1., 61)*2.*np.pi
    y = np.linspace(0., 1., 31)*np.pi
    X, Y = np.meshgrid(x, y)
    X = X.ravel()
    Y = Y.ravel()
    data = np.array([X, Y]).transpose()
    return data, X, Y


class TestDiffusionMap(object):
    @pytest.mark.parametrize('choose_eps', ['fixed', 'bgh'])
    def test_1Dstrip_evals(self, choose_eps):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = 0.25*np.array([1, 4, 9, 16])
        X = np.linspace(0., 1., 81)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.05
        # Setup diffusion map
        eps = 0.005
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, choose_eps=choose_eps, alpha=1.0, k=20)
        mydmap.fit(data)
        test_evals = -4./mydmap.epsilon*(mydmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    @pytest.mark.parametrize('choose_eps', ['fixed', 'bgh'])
    def test_1Dstrip_evecs(self, choose_eps):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        X = np.linspace(0., 1., 81)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.003
        # Setup diffusion map
        eps = 0.005
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, choose_eps=choose_eps, alpha=1.0, k=40)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mydmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    @pytest.mark.parametrize('choose_eps', ['fixed', 'bgh'])
    def test_1Dstrip_nonunif_evals(self, choose_eps):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi with nonuniform sampling.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = 0.25*np.array([1, 4, 9, 16])
        # Setup data and accuracy threshold
        X = (np.linspace(0., 1., 81)**2)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.1
        # Setup diffusion map
        eps = 0.02
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, choose_eps=choose_eps, alpha=1.0, k=40)
        mydmap.fit_transform(data)
        test_evals = -4./mydmap.epsilon*(mydmap.evals - 1)
        print(mydmap.epsilon, test_evals)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    @pytest.mark.parametrize('choose_eps', ['fixed', 'bgh'])
    def test_1Dstrip_nonunif_evecs(self, choose_eps):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi with nonuniform sampling.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        X = (np.linspace(0., 1., 81)**2)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.01
        # Setup diffusion map
        eps = 0.02
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, choose_eps=choose_eps, alpha=1.0, k=40)
        mydmap.fit_transform(data)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mydmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    def test_2Dstrip_evals(self, uniform_2d_data):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = kx^2 + ky^2 for kx = 0.5*[1 0 2 1] and ky = [0 1 0 1].
        real_evals = 0.25*np.array([1, 4, 4, 5])
        # Setup data and accuracy threshold
        data, X, Y = uniform_2d_data
        THRESH = 0.2

        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=100)
        mydmap.fit(data)
        test_evals = -4./mydmap.epsilon*(mydmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.max(errors_eval)
        assert(total_error < THRESH)

    def test_2Dstrip_evecs(self, uniform_2d_data):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(kx*x)*cos(ky*y) for kx = 0.5*[1 0 2 1] and ky = [0 1 0 1].
        # Setup data and accuracy threshold
        data, X, Y = uniform_2d_data
        THRESH = 0.01

        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=100)
        mydmap.fit(data)
        errors_evec = []
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*1*X), mydmap.evecs[:, 0])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(Y), mydmap.evecs[:, 1])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*2*X), mydmap.evecs[:, 2])[0, 1]))
        errors_evec.append(abs(np.corrcoef(np.cos(0.5*1*X)*np.cos(Y), mydmap.evecs[:, 3])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)
        assert(total_error < THRESH)

    def test_sphere_evals(self, spherical_data):
        """
        Test that we compute the correct eigenvalues on a 2d sphere embedded in 3d.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        data, Phi, Theta = spherical_data
        # Setup true values to test against.
        real_evals = np.array([2, 2, 2, 6])  # =l(l+1)
        THRESH = 0.1
        eps = 0.05
        mydmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, alpha=1.0, k=400)
        mydmap.fit(data)
        test_evals = -4./mydmap.epsilon*(mydmap.evals - 1)

        # Check eigenvalues pass below error tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        max_eval_error = np.max(errors_eval)
        assert(max_eval_error < THRESH)

    def test_sphere_evecs(self, spherical_data):
        """
        Test that we compute the correct eigenvectors (spherical harmonics) on a 2d sphere embedded in R^3.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        data, Phi, Theta = spherical_data
        THRESH = 0.001
        eps = 0.05
        mydmap = dm.DiffusionMap(n_evecs=1, epsilon=eps, alpha=1.0, k=400)
        mydmap.fit(data)
        # rotate sphere so that maximum of first DC is at the north pole
        northpole = np.argmax(mydmap.dmap[:, 0])
        phi_n = Phi[northpole]
        theta_n = Theta[northpole]
        R = np.array([[np.sin(theta_n)*np.cos(phi_n), np.sin(theta_n)*np.sin(phi_n), -np.cos(theta_n)],
                      [-np.sin(phi_n), np.cos(phi_n), 0],
                      [np.cos(theta_n)*np.cos(phi_n), np.cos(theta_n)*np.sin(phi_n), np.sin(theta_n)]])
        data_rotated = np.dot(R, data.transpose())
        # check that error is beneath tolerance.
        evec_error = 1 - np.corrcoef(mydmap.dmap[:, 0], data_rotated[2, :])[0, 1]
        assert(evec_error < THRESH)


class TestNystroem(object):
    def test_2Dstrip_nystroem(self, uniform_2d_data):
        """
        Test the nystroem extension in the transform() function.
        """
        # Setup data and accuracy threshold
        data, X, Y = uniform_2d_data
        THRESH = 0.01
        # Setup diffusion map
        eps = 0.01
        mydmap = dm.DiffusionMap(n_evecs=1, epsilon=eps, alpha=1.0, k=100)
        mydmap.fit(data)
        # Setup values to test against (regular grid)
        x_test, y_test = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(0, np.pi, 40))
        X_test = np.array([x_test.ravel(), y_test.ravel()]).transpose()
        # call nystroem extension
        dmap_ext = mydmap.transform(X_test)
        # extract first diffusion coordinate and normalize
        V_test = dmap_ext[:, 0]
        V_test = V_test/np.linalg.norm(V_test)
        # true dominant eigenfunction = cos(0.5*x), normalize
        V_true = np.cos(.5*x_test).ravel()
        V_true = V_true/np.linalg.norm(V_true)
        # compute L2 error, deal with remaining sign ambiguity
        error = min([np.linalg.norm(V_true+V_test), np.linalg.norm(V_true-V_test)])
        assert(error < THRESH)


class TestTMDiffusionMap(object):
    @pytest.mark.parametrize('choose_eps', ['fixed', 'bgh'])
    def test_1Dstrip_evals(self, choose_eps):
        """
        Test that we compute the correct eigenvalues on a 1d strip of length 2*pi.
        Using the data set of uniformly distributed points on 1 dimensional periodic domain [0,2 pi],
        the TMDmap with constat vector equal to one as target density vector
         (i.e. for flat potential exp(-V(q)) with V = 0).
        The TMDmap with constant (equal to one) target-measure gives diffusion maps for alpha=1.0.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvalue approximation will fail if k is set too small, or epsilon not optimal (sensitive).
        """
        # Setup true values to test again.
        # real_evals = k^2 for k in 0.5*[1 2 3 4]
        real_evals = 0.25*np.array([1, 4, 9, 16])
        # Setup data and accuracy threshold
        X = np.linspace(0., 1., 81)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.05
        # Setup diffusion map
        eps = 0.005

        target_distribution = np.ones(len(data))
        mytmdmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, choose_eps=choose_eps, k=100)
        mytmdmap.fit_transform(data, weights=target_distribution)
        test_evals = -4./mytmdmap.epsilon*(mytmdmap.evals - 1)

        # Check that relative error values are beneath tolerance.
        errors_eval = abs((test_evals - real_evals)/real_evals)
        total_error = np.min(errors_eval)
        assert(total_error < THRESH)

    @pytest.mark.parametrize('choose_eps', ['fixed', 'bgh'])
    def test_1Dstrip_evecs(self, choose_eps):
        """
        Test that we compute the correct eigenvectors (cosines) on a 1d strip of length 2*pi.
        The TMDmap with constant (equal to one) target-measure gives diffusion maps for alpha=1.0.
        Diffusion map parameters in this test are hand-selected to give good results.
        Eigenvector approximation will fail if epsilon is set way too small or too large (robust).
        """
        # Setup true values to test again.
        # real_evecs = cos(k*x) for k in 0.5*[1 2 3 4]
        # Setup data and accuracy threshold
        X = np.linspace(0., 1., 81)*2.*np.pi
        data = np.array([X]).transpose()
        THRESH = 0.05
        # Setup diffusion map
        eps = 0.005
        target_distribution = np.ones(len(data))
        mytmdmap = dm.DiffusionMap(n_evecs=4, epsilon=eps, choose_eps=choose_eps, k=100)
        mytmdmap.fit_transform(data, weights=target_distribution)
        errors_evec = []
        for k in np.arange(4):
            errors_evec.append(abs(np.corrcoef(np.cos(0.5*(k+1)*X), mytmdmap.evecs[:, k])[0, 1]))

        # Check that relative error values are beneath tolerance.
        total_error = 1 - np.min(errors_evec)

        assert(total_error < THRESH)


class TestSymmetrization():
    test_mat = csr_matrix([[0, 2.], [0, 3.]])

    def test_and_symmetrization(self):
        ref_mat = np.array([[0, 0], [0, 3.]])
        symmetrized = dm._symmetrize_matrix(self.test_mat, mode='and')
        symmetrized = symmetrized.toarray()
        assert (np.linalg.norm(ref_mat - symmetrized) == 0.)

    def test_or_symmetrization(self):
        ref_mat = np.array([[0, 2.], [2., 3.]])
        symmetrized = dm._symmetrize_matrix(self.test_mat, mode='or')
        symmetrized = symmetrized.toarray()
        assert (np.linalg.norm(ref_mat - symmetrized) == 0.)

    def test_avg_symmetrization(self):
        ref_mat = np.array([[0, 1.], [1., 3.]])
        symmetrized = dm._symmetrize_matrix(self.test_mat, mode='average')
        symmetrized = symmetrized.toarray()
        assert (np.linalg.norm(ref_mat - symmetrized) == 0.)
