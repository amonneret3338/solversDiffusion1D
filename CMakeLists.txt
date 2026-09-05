cmake_minimum_required(VERSION 3.16)
project(MonProjet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(MonProjet
    src/chaleur1D.cpp
    src/solverTridiagonal.cpp
    src/solverGauss.cpp
    src/solverGaussSeidel.cpp
    src/solverCG.cpp

)

target_include_directories(MonProjet
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(MonProjet PRIVATE Eigen3::Eigen)

