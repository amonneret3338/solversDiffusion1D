#include <iostream>
#include "../include/solverTridiagonal.hpp"

ThomasSolver::ThomasSolver(std::vector<double> const& a,
                           std::vector<double> const& b,
                           std::vector<double> const& c)
    : a_(a), b_(b), c_(c), size_(b.size())
{
    if (a_.size() != c_.size() || b_.size() != size_) {
        throw std::runtime_error(
            "ThomasSolver -> a, b, c must have the same size"
        );
    }
    if (size_ == 0) {
        throw std::runtime_error(
            "ThomasSolver -> system cannot be empty"
        );
    }
    cPrime_.resize(size_);
    // Premier élément
    if (std::abs(b_[0]) <= pivotEpsilon) {
        throw std::runtime_error(
            "ThomasSolver -> zero pivot at index 0"
        );
    }
    cPrime_[0] = c_[0] / b_[0];
    // Forward sweep pour c'
    for (size_t i = 1; i < size_; ++i) {
        double pivot = b_[i] - a_[i] * cPrime_[i - 1];
        if (std::abs(pivot) <= pivotEpsilon) {
            throw std::runtime_error(
                "ThomasSolver -> zero pivot"
            );
        }
        cPrime_[i] = c_[i] / pivot;
    }
}

std::vector<double> ThomasSolver::solve(
    std::vector<double> const& d) const
{
    if (d.size() != size_) {
        throw std::runtime_error(
            "ThomasSolver::solve -> d has wrong size"
        );
    }
    std::vector<double> dPrime(size_);
    std::vector<double> solution(size_);
    // Forward sweep pour d'
    if (std::abs(b_[0]) <= pivotEpsilon) {
        throw std::runtime_error(
            "ThomasSolver::solve -> zero pivot at index 0"
        );
    }
    dPrime[0] = d[0] / b_[0];
    for (size_t i = 1; i < size_; ++i) {
        double pivot = b_[i] - a_[i] * cPrime_[i - 1];
        if (std::abs(pivot) <= pivotEpsilon) {
            throw std::runtime_error(
                "ThomasSolver::solve -> zero pivot"
            );
        }
        dPrime[i] = (d[i] - a_[i] * dPrime[i - 1]) / pivot;
    }
    // Back substitution
    solution[size_ - 1] = dPrime[size_ - 1];
    for (size_t i = size_ - 1; i > 0; --i) {
        solution[i - 1] =
            dPrime[i - 1] - cPrime_[i - 1] * solution[i];
    }
    return solution;
}
