add_library(maskgen SHARED
    mask_generator.cpp
    mesh.cpp
    vulkan_context.cpp
)

target_include_directories(maskgen
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(maskgen
    PRIVATE
        Vulkan::Vulkan
        ${OpenCV_LIBS}
        glm::glm
        shaderc
)

set_target_properties(maskgen PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

install(TARGETS maskgen
    EXPORT maskgenTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
