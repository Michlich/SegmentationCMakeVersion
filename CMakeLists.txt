# CMakeList.txt: файл проекта CMake верхнего уровня; проведите глобальную конфигурацию
# и включите сюда подпроекты.
#
cmake_minimum_required (VERSION 3.8)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(SegmentationProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
# --- Поиск внешних зависимостей (Qt и OpenCV) ---
# Их по-прежнему лучше искать в системе или через CMAKE_PREFIX_PATH.
# Это стандартная практика.
# Пример вызова: cmake -S . -B build -D CMAKE_PREFIX_PATH="C:/Qt/5.15.2/msvc2019_64;E:/opencv/build"

      
# --- НАЧАЛО ДИАГНОСТИЧЕСКОГО БЛОКА ---
#message(STATUS "========== ЗАПУСКАЕМ ПОИСК ПАКЕТОВ В РЕЖИМЕ ДИАГНОСТИКИ ==========")
#set(CMAKE_FIND_DEBUG_MODE TRUE)
# --- КОНЕЦ ДИАГНОСТИЧЕСКОГО БЛОКА ---

# Теперь идут ваши обычные команды
#find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)
#find_package(Qt5 REQUIRED COMPONENTS Widgets)

# --- НАЧАЛО ВТОРОГО ДИАГНОСТИЧЕСКОГО БЛОКА ---
#set(CMAKE_FIND_DEBUG_MODE FALSE) # Выключаем режим, чтобы не засорять вывод
#message(STATUS "========== ДИАГНОСТИКА ЗАВЕРШЕНА ==========")
# --- КОНЕЦ ВТОРОГО ДИАГНОСТИЧЕСКОГО БЛОКА ---


#set(QT_BIN "C:/Qt/Qt5.12.12/5.12.12/msvc2017_64/bin")
#set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs) 
find_package(Qt5 REQUIRED COMPONENTS Widgets)



set(ONNX_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/packages/Microsoft.ML.OnnxRuntime.1.22.0")
set(ONNX_INCLUDE_DIR "${ONNX_VENDOR_DIR}/build/native/include")

# 2. Определяем платформу и пути к файлам
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ONNX_RUNTIME_DIR "${ONNX_VENDOR_DIR}/runtimes/win-x64/native")
    else()
        message(FATAL_ERROR "Только 64-битные сборки поддерживаются на Windows для ONNX Runtime в этом проекте.")
    endif()

    set(ONNX_LIB "${ONNX_RUNTIME_DIR}/onnxruntime.lib")
    set(ONNX_PROVIDERS_LIB "${ONNX_RUNTIME_DIR}/onnxruntime_providers_shared.lib")
    set(ONNX_RUNTIME_DLLS
        "${ONNX_RUNTIME_DIR}/onnxruntime.dll"
        "${ONNX_RUNTIME_DIR}/onnxruntime_providers_shared.dll"
    )
    set(ONNX_TARGET_LIBS onnxruntime::onnxruntime onnxruntime::providers_shared)
    

elseif(UNIX AND NOT APPLE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ONNX_RUNTIME_DIR "${ONNX_VENDOR_DIR}/runtimes/linux-x64/native")
    else()
        message(FATAL_ERROR "Только 64-битные сборки поддерживаются на Linux для ONNX Runtime в этом проекте.")
    endif()
    
    # В Linux обычно один .so файл
    file(GLOB ONNX_SO_FILE "${ONNX_RUNTIME_DIR}/libonnxruntime.so*")
    set(ONNX_LIB ${ONNX_SO_FILE})
    set(ONNX_RUNTIME_DLLS ${ONNX_SO_FILE}) # Для копирования
    set(ONNX_TARGET_LIBS onnxruntime::onnxruntime)

else()
    message(FATAL_ERROR "Неподдерживаемая платформа для ONNX Runtime: ${CMAKE_SYSTEM_NAME}")
endif()

# 3. Создаем импортированные таргеты для ONNX Runtime
add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNX_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_INCLUDE_DIR}"
)
if(WIN32)
    # Providers lib есть только в Windows дистрибутиве
    add_library(onnxruntime::providers_shared UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::providers_shared PROPERTIES
        IMPORTED_LOCATION "${ONNX_PROVIDERS_LIB}"
    )
endif()


# --- Создание исполняемого файла ---
add_executable(SegmentationProject
    "main.cpp"
    "Segmentcl.cpp"
    "Segmentcl.h"
    "Segmentcl.qrc"
    "Segmentcl.ui"
)

# --- Настройка таргета ---
if(MSVC)
    target_compile_definitions(SegmentationProject PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

# Линкуем библиотеки. Список ONNX_TARGET_LIBS был сформирован выше.
target_link_libraries(SegmentationProject PRIVATE
    Qt5::Widgets
    onnxruntime::onnxruntime
    onnxruntime::providers_shared
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
)
# --- Копирование DLL/SO для запуска из папки сборки (POST_BUILD) ---
# Это удобно для разработки и отладки

# Копируем DLL/SO от ONNX Runtime
foreach(RUNTIME_FILE ${ONNX_RUNTIME_DLLS})
     add_custom_command(TARGET SegmentationProject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${RUNTIME_FILE}"
        "$<TARGET_FILE_DIR:SegmentationProject>"
        COMMENT "Copying runtime dependency ${RUNTIME_FILE}"
    )
endforeach()

# Для Windows запускаем windeployqt, чтобы скопировать Qt DLLs
if(WIN32)
    if(TARGET Qt5::windeployqt)
    # Современный способ для Qt >= 5.15
        add_custom_command(TARGET SegmentationProject POST_BUILD
        COMMAND $<TARGET_FILE:Qt5::windeployqt>
            --dir "$<TARGET_FILE_DIR:SegmentationProject>"
            --pdb
            $<TARGET_FILE_NAME:SegmentationProject>
        COMMENT "Running windeployqt (modern method) to copy Qt DLLs"
    )
    else()
    # Старый, но надежный способ для Qt < 5.15
    find_package(Qt5 COMPONENTS Core)
    get_target_property(QT_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)

    add_custom_command(TARGET SegmentationProject POST_BUILD
        COMMAND "${QT_BIN_DIR}/windeployqt.exe"
            --dir "$<TARGET_FILE_DIR:SegmentationProject>"
            --pdb
            "$<TARGET_FILE:SegmentationProject>"  # <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
        COMMENT "Running windeployqt (legacy method) to copy Qt DLLs"
    )
endif()


    # --- НАЧАЛО ИСПРАВЛЕННОГО БЛОКА КОПИРОВАНИЯ OPENCV DLL ---

    # Находим путь к папке с бинарниками OpenCV
    set(OPENCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin") # <-- Исправлено на vc16

    # Ищем отладочные DLL (с суффиксом 'd')
    file(GLOB OPENCV_DEBUG_DLLS "${OPENCV_BIN_DIR}/*d.dll")
    # Ищем релизные DLL (без суффикса 'd')
    file(GLOB OPENCV_RELEASE_DLLS_ALL "${OPENCV_BIN_DIR}/*.dll")
    # Удаляем отладочные из общего списка, чтобы остались только релизные
    list(REMOVE_ITEM OPENCV_RELEASE_DLLS_ALL ${OPENCV_DEBUG_DLLS})
    set(OPENCV_RELEASE_DLLS ${OPENCV_RELEASE_DLLS_ALL})


    # Команда для копирования отладочных DLL только для Debug сборки
    add_custom_command(TARGET SegmentationProject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_DEBUG_DLLS}
        $<TARGET_FILE_DIR:SegmentationProject>
        COMMENT "Copying OpenCV Debug DLLs"
        CONFIGURATIONS Debug
    )

    # Команда для копирования релизных DLL только для Release сборок
    add_custom_command(TARGET SegmentationProject POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENCV_RELEASE_DLLS}
        $<TARGET_FILE_DIR:SegmentationProject>
        COMMENT "Copying OpenCV Release DLLs"
        CONFIGURATIONS Release RelWithDebInfo MinSizeRel
    )

    # --- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА ---
endif()

# --- Настройка установки (для создания дистрибутива) ---
# Команда `cmake --install build` соберет все в одну папку

# 1. Устанавливаем наш исполняемый файл
install(TARGETS SegmentationProject RUNTIME DESTINATION bin)

# 2. Устанавливаем рантайм-зависимости
install(FILES ${ONNX_RUNTIME_DLLS} DESTINATION bin)

if(WIN32)
    # Здесь можно добавить правила для установки DLL от Qt и OpenCV
    # Это более сложная тема, но для начала копирования в папку сборки достаточно.
endif()

if(UNIX)
    # Устанавливаем RPATH, чтобы исполняемый файл искал .so в той же папке
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
