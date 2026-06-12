find_package(nlohmann_json REQUIRED)

add_executable(benchmark_maskgen benchmark_maskgen.cpp)

target_compile_definitions(benchmark_maskgen
    PRIVATE
        BENCHMARK_MESH_PATH="${CMAKE_SOURCE_DIR}/model/ad.step"
        BENCHMARK_CAMERA_PATH="${CMAKE_SOURCE_DIR}/camera_info.json"
)

target_link_libraries(benchmark_maskgen
    PRIVATE
        maskgen
        ${OpenCV_LIBS}
        Catch2::Catch2WithMain
        nlohmann_json::nlohmann_json
)

include(Catch)
catch_discover_tests(benchmark_maskgen)
