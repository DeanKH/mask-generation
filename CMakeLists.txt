cmake_minimum_required(VERSION 3.20)
project(maskgen VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

find_path(OpenCASCADE_INCLUDE_DIR
    NAMES Standard.hxx
    PATH_SUFFIXES opencascade
)

if(NOT OpenCASCADE_INCLUDE_DIR)
    message(FATAL_ERROR "OpenCASCADE headers not found. Install libocct-foundation-dev.")
endif()

set(_OCCT_LIBS
    TKernel TKMath TKBRep TKGeomBase TKGeomAlgo TKTopAlgo
    TKShHealing TKXSBase TKSTEPBase TKSTEP209 TKSTEP TKMesh
)
set(OpenCASCADE_LIBRARIES "")
foreach(_lib ${_OCCT_LIBS})
    find_library(OCCT_LIB_${_lib} ${_lib})
    if(NOT OCCT_LIB_${_lib})
        message(FATAL_ERROR "OpenCASCADE library ${_lib} not found")
    endif()
    list(APPEND OpenCASCADE_LIBRARIES ${OCCT_LIB_${_lib}})
endforeach()
message(STATUS "Found OpenCASCADE: ${OpenCASCADE_INCLUDE_DIR}")

add_subdirectory(src)
add_subdirectory(app)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.0
)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(tests)
enable_testing()

include(GNUInstallDirs)

install(DIRECTORY include/maskgen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT maskgenTargets
    FILE maskgenTargets.cmake
    NAMESPACE maskgen::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/maskgen
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/maskgenConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/maskgenConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/maskgen
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/maskgenConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/maskgenConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/maskgenConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/maskgen
)
