cmake_minimum_required(VERSION 3.10)
project(teaser_register CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(PCL 1.8 REQUIRED COMPONENTS common io filters features registration kdtree)
find_package(teaserpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(teaser_register teaser_register.cc)

target_link_libraries(teaser_register
    ${PCL_LIBRARIES}
    teaserpp::teaser_registration
    teaserpp::teaser_features
    Eigen3::Eigen
)