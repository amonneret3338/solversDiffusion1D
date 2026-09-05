#pragma once
#include <Eigen/Dense>
#include <cstddef>

class ConjugateGradientSolver {
public:
    ConjugateGradientSolver(
        Eigen::MatrixXd const& A,
        double const& tolerance,
        size_t const& maxIter
    );

    // Résout A*x = b, en partant du point initial x0.
    Eigen::VectorXd solve(
        Eigen::VectorXd const& x0,
        Eigen::VectorXd const& b
    ) const;

private:
    Eigen::MatrixXd A_;
    size_t size_;
    double tol_;
    size_t maxIterations_;
};
