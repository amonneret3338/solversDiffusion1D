// Solver Gauss-Seidel
#include "../include/solverGaussSeidel.hpp"

#include <stdio.h>
#include <iostream>
#include <stdexcept>

GaussSeidelSolver::GaussSeidelSolver(
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
            "GaussSeidelSolver -> matrix must be square"
        );
    }

    if (tolerance <= 0.0) {
        throw std::runtime_error(
            "GaussSeidelSolver -> tolerance must be positive"
        );
    }

    if (maxIter == 0) {
        throw std::runtime_error(
            "GaussSeidelSolver -> maxIter must be positive"
        );
    }

    // Vérification de la diagonale
    for (size_t i = 0; i < size_; ++i) {
        if (A_(i, i) == 0.0) {
            throw std::runtime_error(
                "GaussSeidelSolver -> zero diagonal element"
            );
        }
    }
}

Eigen::VectorXd GaussSeidelSolver::solve(
    Eigen::VectorXd const& x0,
    Eigen::VectorXd const& b
) const
{
    if (b.size() != static_cast<Eigen::Index>(size_)) {
        throw std::runtime_error(
            "GaussSeidelSolver::solve -> b has wrong size"
        );
    }

    if (x0.size() != static_cast<Eigen::Index>(size_)) {
        throw std::runtime_error(
            "GaussSeidelSolver::solve -> x0 has wrong size"
        );
    }

    Eigen::VectorXd x = x0;

    for (size_t k = 0; k < maxIterations_; ++k) {

        Eigen::VectorXd xOld = x;

        for (size_t i = 0; i < size_; ++i) {

            double sum = 0.0;

            // Valeurs déjà mises à jour : Gauss-Seidel
            for (size_t j = 0; j < i; ++j) {
                sum += A_(i, j) * x(j);
            }

            // Valeurs de l'ancienne itération
            for (size_t j = i + 1; j < size_; ++j) {
                sum += A_(i, j) * xOld(j);
            }

            x(i) = (b(i) - sum) / A_(i, i);
        }

        // Critère d'arrêt :
        // ||x - xOld|| / ||x||
        double xNorm = x.norm();

        if (xNorm == 0.0) {
            xNorm = 1.0;
        }

        double residue = (x - xOld).norm() / xNorm;

        if (residue < tol_) {
            return x;
        }
    }

    std::cout
        << "GaussSeidel reached maximum number of iterations"
        << std::endl;

    return x;
}

