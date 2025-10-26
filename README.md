# SegmentationCMakeVersion
Для работы программы необходимы:
1. фреймворк Qt весрии 5.12.12;
2. библиотека OpenCV версии 4.11.0;
3. на Linux может понадобиться дополнительная установка onnxruntime версии 1.22.0.
Команды для сборки windows (рекомендуемо использования компилятора msvc):
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T v142 -D Qt5_DIR="C:/path/to/Qt/Qt5.12.12/5.12.12/msvc2017_64/lib/cmake/Qt5" -D OpenCV_DIR="C:/path/to/openCV/opencv/build"
cmake --build build --config Debug
Для linux:
cmake -S . -B build -D Qt5_DIR="/path/to/qt5/lib/cmake/Qt5" -D OpenCV_DIR="/path/to/opencv/build"
cmake --build build --config Debug
После сборки проекта необходимо скопировать .onnx файлы в директорию с исполняемым файлом.

При запуске программы необходимо выбрать файл PNG или BMP размером СТРОГО 512X512 пикселей.
