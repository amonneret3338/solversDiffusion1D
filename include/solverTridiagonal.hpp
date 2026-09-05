#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>


struct TridiagonalSystem {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> d;
};

class ThomasSolver {
public:
    ThomasSolver(std::vector<double> const& a,
                 std::vector<double> const& b,
                 std::vector<double> const& c);

    std::vector<double> solve(std::vector<double> const& d) const;

private:
    static constexpr double pivotEpsilon = 1.e-12;

    std::vector<double> a_;
    std::vector<double> b_;
    std::vector<double> c_;
    std::vector<double> cPrime_;
    size_t size_;
};


