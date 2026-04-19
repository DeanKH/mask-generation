add_executable(maskgen_cli main.cpp)

target_link_libraries(maskgen_cli
    PRIVATE
        maskgen
        ${OpenCV_LIBS}
)

install(TARGETS maskgen_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
