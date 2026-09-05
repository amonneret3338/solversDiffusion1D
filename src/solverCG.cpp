#include "../include/solverCG.hpp"
#include <iostream>
#include <stdexcept>

ConjugateGradientSolver::ConjugateGradientSolver(
    Eigen::MatrixXd const& A,
    double const& tolerance,
    size_t const& maxIter
)
    : A_(A),
      size_(A.rows()),
      tol_(tolerance),
      maxIterations_(maxIter)
{
    if (A.rows() != A.cols()) {
        throw std::runtime_error(
            "ConjugateGradientSolver -> matrix must be square"
        );
    }
    if (tolerance <= 0.0) {
        throw std::runtime_error(
            "ConjugateGradientSolver -> tolerance must be positive"
        );
    }
    if (maxIter == 0) {
        throw std::runtime_error(
            "ConjugateGradientSolver -> maxIter must be positive"
        );
    }
    // Vérification (nécessaire mais non suffisante) de symétrie.
    // Une vraie vérification de "définie positive" coûterait cher (valeurs propres) ;
    // on se contente ici d'un garde-fou sur la symétrie.
    if (!A_.isApprox(A_.transpose(), 1e-10)) {
        throw std::runtime_error(
            "ConjugateGradientSolver -> matrix must be symmetric"
        );
    }
}

Eigen::VectorXd ConjugateGradientSolver::solve(
    Eigen::VectorXd const& x0,
    Eigen::VectorXd const& b
) const
{
    if (b.size() != static_cast<Eigen::Index>(size_)) {
        throw std::runtime_error(
            "ConjugateGradientSolver::solve -> b has wrong size"
        );
    }
    if (x0.size() != static_cast<Eigen::Index>(size_)) {
        throw std::runtime_error(
            "ConjugateGradientSolver::solve -> x0 has wrong size"
        );
    }

    Eigen::VectorXd x = x0;
    Eigen::VectorXd r = b - A_ * x;
    Eigen::VectorXd p = r;

    double residualNormInitial = r.norm();
    if (residualNormInitial == 0.0) {
        residualNormInitial = 1.0;  // évite une division par zéro si x0 est déjà solution
    }

    double rDotR = r.dot(r);

    for (size_t k = 0; k < maxIterations_; ++k) {
        Eigen::VectorXd Ap = A_ * p;   // seul produit matrice-vecteur de l'itération

        double const alpha = rDotR / p.dot(Ap);

        x += alpha * p;
        Eigen::VectorXd rNew = r - alpha * Ap;

        double const relativeResidual = rNew.norm() / residualNormInitial;
        //std::cout << "CG residual (k=" << k << ") : " << relativeResidual << std::endl;

        if (relativeResidual < tol_) {
            return x;
        }

        double const rNewDotRNew = rNew.dot(rNew);
        double const beta = rNewDotRNew / rDotR;

        p = rNew + beta * p;
        r = rNew;
        rDotR = rNewDotRNew;
    }

    std::cout << "ConjugateGradient reached maximum number of iterations" << std::endl;
    return x;
}
