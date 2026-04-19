cmake_minimum_required(VERSION 3.20)
project(maskgen VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(src)
add_subdirectory(app)

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
