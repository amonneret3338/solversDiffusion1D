// This function is expected to simulate the heat transfer equation in 1D.
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "../include/solverTridiagonal.hpp"
#include "../include/solverGauss.hpp"
#include "../include/solverGaussSeidel.hpp"
#include "../include/solverCG.hpp"

class Timer {
public:
    void tic() {
        start = std::chrono::high_resolution_clock::now();
    }

    double toc() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

double const dx = 1.;
double const alpha = 10;
double const dt = 0.5 * dx * dx / alpha;  // limit is frac{dx^2}{2 alpha}
double const tempLeft = 200.;
double const tempRight = 300.;
double const tempInit = 250.;
double const precision = 1.e-10;
double const r = alpha * dt / (dx * dx);

std::vector<double> createTemperatureVector(double const nbP, double const TL, double const TR, double const Tmid)
{
    std::vector<double> temperatures(nbP, 0.);
    temperatures[0] = TL;
    temperatures[nbP - 1] = TR;
    size_t nbElements = temperatures.size();
    for (size_t i = 1; i < nbElements - 1; i++) {
        temperatures[i] += Tmid;
    }
    return temperatures;
}

double computeDifference(std::vector<double> const& temp1, std::vector<double> const& temp2)
{
    size_t nbElements = temp1.size();
    if (nbElements != temp2.size()) {
        throw std::runtime_error("computeDifference -> the number of elements of the two vector is not the same");
    }

    double error = 0.;
    for (size_t i = 0; i < nbElements; i++) {
        error += (temp1[i] != 0.) ? std::abs((temp1[i] - temp2[i]) / temp1[i]) : std::abs((temp1[i] - temp2[i]) / 1.);
    }
    return error;
}

void printVector(std::vector<double> const& myVector)
{
    size_t nbElements = myVector.size();
    for (size_t i = 0; i < nbElements; i++) {
        std::cout << myVector[i] << " ";
    }
    std::cout << std::endl;
}

// Construit la matrice dense (n x n) à partir des trois diagonales,
// nécessaire pour GaussSolver (Eigen travaille sur des matrices pleines).
Eigen::MatrixXd buildDenseMatrix(std::vector<double> const& a,
                                  std::vector<double> const& b,
                                  std::vector<double> const& c)
{
    size_t const n = b.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);

    for (size_t i = 0; i < n; ++i) {
        A(i, i) = b[i];
        if (i > 0)     A(i, i - 1) = a[i];
        if (i < n - 1) A(i, i + 1) = c[i];
    }
    return A;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./MonProjet <size>\n";
        return 1;
    }

    const size_t size = std::stoul(argv[1]);
    std::cout << "size = " << size << std::endl;

    if (size < 4) {
        throw std::runtime_error("main -> the number of points for the calculation is not high enough");
    }

    Timer timer;

    std::vector<double> tempInitial = createTemperatureVector(size, tempLeft, tempRight, tempInit);

    const size_t interiorSize = size - 2;

    // Definition of the three diagonals (identique pour Thomas et Gauss)
    std::vector<double> a(interiorSize, -r);
    std::vector<double> b(interiorSize, 1. + 2. * r);
    std::vector<double> c(interiorSize, -r);

    // The first equation has no T_0 unknown
    a[0] = 0.;
    // The last equation has no T_(N-1) unknown
    c[interiorSize - 1] = 0.;

    double epsilon;
    size_t counter;
    std::vector<double> tempNew;


    // Useful for everyone (Gauss and GaussSeidel)
    Eigen::MatrixXd A = buildDenseMatrix(a, b, c);


    // ---------------------------------------------------------
    // Résolution avec ThomasSolver
    // ---------------------------------------------------------
    ThomasSolver thomasSolver(a, b, c);

    std::vector<double> tempOldThomas(
        tempInitial.begin() + 1,
        tempInitial.end() - 1
    );

    epsilon = 0.1;
    counter = 0;
    timer.tic();
    while (epsilon > precision) {
        std::vector<double> rhs = tempOldThomas;
        rhs[0] += r * tempLeft;
        rhs[interiorSize - 1] += r * tempRight;

        tempNew = thomasSolver.solve(rhs);
        epsilon = computeDifference(tempOldThomas, tempNew);
        tempOldThomas = tempNew;
        ++counter;
    }
    std::cout << "Size " << size << " Elapsed with Thomas method: " << timer.toc() << " s\n";
    // ---------------------------------------------------------
    // Résolution avec GaussSolver (Eigen, LU dense) AI written
    // ---------------------------------------------------------
    GaussSolver gaussSolver(A);

    std::vector<double> tempOldGauss(
        tempInitial.begin() + 1,
        tempInitial.end() - 1
    );

    epsilon = 0.1;
    counter = 0;
    timer.tic();
    while (epsilon > precision) {
        std::vector<double> rhs = tempOldGauss;
        rhs[0] += r * tempLeft;
        rhs[interiorSize - 1] += r * tempRight;

        tempNew = gaussSolver.solve(rhs);
        epsilon = computeDifference(tempOldGauss, tempNew);
        tempOldGauss = tempNew;
        ++counter;
    }
    std::cout << "Size " << size << " Elapsed with Gauss  method: " << timer.toc() << " s\n";

    // ---------------------------------------------------------
    // Résolution avec GaussSeidelSolver
    // ---------------------------------------------------------
    GaussSeidelSolver gaussSeidelSolver(
        A,
        1.e-10,
        1000
    );
    
    // Initialisation de la solution
    Eigen::VectorXd x0(interiorSize);
   
    for (int i = 0; i < interiorSize; ++i) {
        x0(i) = tempInitial[i + 1];
    }
    
//    Eigen::VectorXd tempOldGaussEigen = x0;
    Eigen::VectorXd newTemperature = x0;
    Eigen::VectorXd oldTemperature = x0;
    
    epsilon = 0.1;
    counter = 0;
    
    
    timer.tic();
    while (epsilon > precision) {
    
        // Second membre
        Eigen::VectorXd rhs = oldTemperature;
    
        rhs(0) += r * tempLeft;
        rhs(interiorSize - 1) += r * tempRight;
    
        // Résolution A * x = rhs
        newTemperature = gaussSeidelSolver.solve(
            oldTemperature,
            rhs
        );
    
        // Critère de convergence de la boucle extérieure
        epsilon = (oldTemperature - newTemperature).norm() / oldTemperature.norm();
    
        oldTemperature = newTemperature;
    
        ++counter;
    }
    
    std::cout
        << "Size " << size
        << " Elapsed with GausSe method: "
        << timer.toc()
        << " s\n";

    // ---------------------------------------------------------
    // Résolution avec ConjugateGradientSolver
    // ---------------------------------------------------------
    ConjugateGradientSolver cgSolver(
        A,
        1.e-10,
        1000
    );
    
    Eigen::VectorXd oldTemperatureCG = x0;   // x0 déjà construit plus haut pour Gauss-Seidel
    Eigen::VectorXd newTemperatureCG(interiorSize);
    
    epsilon = 0.1;
    counter = 0;
    
    timer.tic();
    while (epsilon > precision) {
        Eigen::VectorXd rhs = oldTemperatureCG;
        rhs(0) += r * tempLeft;
        rhs(interiorSize - 1) += r * tempRight;
    
        newTemperatureCG = cgSolver.solve(oldTemperatureCG, rhs);
    
        epsilon = (oldTemperatureCG - newTemperatureCG).norm() / oldTemperatureCG.norm();
        oldTemperatureCG = newTemperatureCG;
    
        ++counter;
    }
    std::cout << "Size " << size << " Elapsed with CG method: " << timer.toc() << " s\n";

    // ---------------------------------------------------------
    // Vérification de cohérence : les deux méthodes doivent converger
    // vers (quasi) la même solution.
    // ---------------------------------------------------------
    double crossCheck = computeDifference(tempOldThomas, tempOldGauss);
    std::cout << "Cross-check Thomas vs Gauss (should be ~0): " << crossCheck << std::endl;

    Eigen::VectorXd solutionThomas = Eigen::Map<Eigen::VectorXd>(tempOldThomas.data(), tempOldThomas.size());

    double const difference = (solutionThomas- oldTemperature).norm() /solutionThomas.norm();
    std::cout << "Cross-check Thomas vs GauSe (should be ~0): " <<difference << std::endl;

    double const differenceCG = (solutionThomas- oldTemperatureCG).norm() /solutionThomas.norm();
    std::cout << "Cross-check Thomas vs CG    (should be ~0): " <<differenceCG << std::endl;


    return 0;
}
