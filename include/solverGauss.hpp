// solverGauss.hpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

class GaussSolver {
public:
    // Prend la matrice pleine A (dense) — construite une seule fois,
    // la factorisation LU est précalculée ici.
    explicit GaussSolver(Eigen::MatrixXd const& A);

    std::vector<double> solve(std::vector<double> const& d) const;

private:
    size_t size_;
    Eigen::PartialPivLU<Eigen::MatrixXd> luDecomposition_;
};
