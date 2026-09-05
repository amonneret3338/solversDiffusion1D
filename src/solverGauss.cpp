// solverGauss.cpp AI written
#include "../include/solverGauss.hpp"

GaussSolver::GaussSolver(Eigen::MatrixXd const& A)
    : size_(static_cast<size_t>(A.rows()))
{
    if (A.rows() != A.cols()) {
        throw std::runtime_error("GaussSolver -> matrix must be square");
    }
    // La factorisation LU (avec pivoting partiel) est calculée ici,
    // une seule fois — coût O(n^3), fait une seule fois pour toute la boucle en temps.
    luDecomposition_ = A.partialPivLu();
}

std::vector<double> GaussSolver::solve(std::vector<double> const& d) const
{
    if (d.size() != size_) {
        throw std::runtime_error("GaussSolver::solve -> d has wrong size");
    }

    // Conversion std::vector -> Eigen::VectorXd
    Eigen::VectorXd rhs = Eigen::Map<Eigen::VectorXd const>(d.data(), d.size());

    // Résolution via L puis U (deux résolutions triangulaires, O(n^2))
    Eigen::VectorXd solutionEigen = luDecomposition_.solve(rhs);

    // Conversion retour Eigen::VectorXd -> std::vector
    return std::vector<double>(
        solutionEigen.data(),
        solutionEigen.data() + solutionEigen.size()
    );
}
