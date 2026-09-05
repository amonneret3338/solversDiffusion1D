// Solver Gauss-Seidel
#pragma once

#include <Eigen/Dense>
#include <cstddef>

class GaussSeidelSolver {
public:
    explicit GaussSeidelSolver(
        Eigen::MatrixXd const& A,
        double const& tolerance,
        std::size_t const& maxIter
    );

    Eigen::VectorXd solve(
        Eigen::VectorXd const& x0,
        Eigen::VectorXd const& b
    ) const;

private:
    Eigen::MatrixXd A_;
    std::size_t size_;
    double tol_;
    std::size_t maxIterations_;
};

