#include "Segmentcl.h"
void Write_JSON_Mask(const cv::Mat& mask, const std::string& filename) {
    std::vector < std::string> nameclass = {"NONE", "building", "woodland", "water", "roads"};
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла" << std::endl;
        return;
    }

    // Записываем начало JSON объекта
    file << "{\n";

    // Проходим по всем классам
    for (int classId = 1; classId <= 4; classId++) {
        // Создаем маску для текущего класса
        cv::Mat classMask = cv::Mat::zeros(mask.size(), CV_8UC1);

        // Заполняем маску
        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                if (mask.at<int>(y, x) == classId) {
                    classMask.at<uchar>(y, x) = 255;
                }
            }
        }
        // Находим контуры
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(classMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Форматируем класс в JSON
        std::stringstream classStream;
        classStream << "\"" << nameclass[classId] << "\": [\n";

        for (size_t i = 0; i < contours.size(); i++) {
            classStream << "    [\n";
            for (size_t j = 0; j < contours[i].size(); j++) {
                classStream << "        {\"x\": " << contours[i][j].x << ", \"y\": " << contours[i][j].y << "}";
                if (j < contours[i].size() - 1) classStream << ",\n";
            }
            classStream << "\n    ]";
            if (i < contours.size() - 1) classStream << ",\n";
        }
        classStream << "\n];\n";

        // Записываем в файл
        file << classStream.str();
    }

    // Закрываем JSON объект
    file << "}";
    file.close();
}

template<typename StringType>
const ORTCHAR_T* convertToOrtsPath(const StringType& str) {
#ifdef _WIN32
    // На Windows просто возвращаем указатель на wchar_t
    return reinterpret_cast<const ORTCHAR_T*>(str.c_str());
#else
    // На других платформах конвертируем в UTF-8
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(str).c_str();
#endif
}
MainWindow::MainWindow(QWidget* parent) : QWidget(parent) {
    QPushButton* openButton = new QPushButton("Open file (.png, .bmp)", this);
    imageLabel = new QLabel(this);
    imageLabel->setFixedSize(450, 450);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setText("None");
    helloLabel = new QLabel(QString::fromUtf8(u8"Итоговые изображения сохраняются в 3-х файлах:\n   - segmented_output_segformer - результат модели SegFormer\n   - segmented_output_unet - результат модели U-Net\n   - segmented_output_2 - результат совмещения двух моделей"), this);
    QPushButton* segButton = new QPushButton("Segment", this);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(openButton);
    layout->addWidget(imageLabel);
    layout->addWidget(helloLabel);
    layout->addWidget(segButton);

    setLayout(layout);

    connect(openButton, &QPushButton::clicked, this, &MainWindow::openFile);
    connect(segButton, &QPushButton::clicked, this, &MainWindow::segFile);
}

void MainWindow::openFile() {
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open file"), QDir::homePath(), tr("Images (*.png *.bmp);;All Files (*)"));
    if (!fileName.isEmpty()) {
        QPixmap pixmap(fileName);
        if (pixmap.isNull()) {
            QMessageBox::warning(this, tr(u8"Ошибка"), tr(u8"Не удалось загрузить изображение."));
            return;
        }
        imageLabel->setPixmap(pixmap.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        SelectFile = fileName;
    }
}
void MainWindow::segFile() {
    imageLabel->setText("Start");

    std::string image_path = SelectFile.toStdString(); // Путь к изображению
    // Пути к моделям Segformer и Unet
    std::wstring model_path_segformer =L"segformer_model.onnx";
    std::wstring model_path_unet = L"unet_model.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceApp");

    Ort::SessionOptions session_options;

    Ort::Session session_segformer(env, convertToOrtsPath(model_path_segformer), session_options);

    imageLabel->setText(u8"Обработка Segformer...");

    cv::Mat input_image_segformer = preprocess_image(image_path, cv::Size(512, 512));
    if (!input_image_segformer.empty()) {
        //разделяем инференс на 2 части, поскольку нам необходимо сохранить значение тензора для совмещения
        std::vector<float> prom_mask_seg = run_inference(session_segformer, input_image_segformer);
        cv::Mat mask_segformer = inference_end(prom_mask_seg.data());

        cv::resize(mask_segformer, mask_segformer, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
        Write_JSON_Mask(mask_segformer, "Segformer_Photo.json");
        cv::Mat color_mask_segformer = decode_segmentation_map(mask_segformer);

        cv::imwrite("segmented_output_segformer.png", color_mask_segformer);

        imageLabel->setText(u8"Маска Segformer сохранена");

        // Загрузка Unet
        Ort::Session session_unet(env, convertToOrtsPath(model_path_unet), session_options);

        imageLabel->setText(u8"Обработка U-Net...");

        cv::Mat input_image_unet = preprocess_image(image_path, cv::Size(512, 512));
        if (!input_image_unet.empty()) {
            std::vector<float> prom_mask_unet = run_inference(session_unet, input_image_unet);
            cv::Mat mask_unet = inference_end(prom_mask_unet.data());
            Write_JSON_Mask(mask_unet, "Unet_Photo.json");

            cv::Mat color_mask_unet = decode_segmentation_map(mask_unet);

            cv::imwrite("segmented_output_unet.png", color_mask_unet);

            imageLabel->setText("Маска U-Net сохранена");


            std::vector<float> prom_mask_2(512 * 512 * 5);
            //Совмещение масок
            Interpolate2(prom_mask_seg, prom_mask_unet, prom_mask_2);
            cv::Mat mask_2 = inference_end(prom_mask_2.data());

            cv::resize(mask_2, mask_2, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST);
            Write_JSON_Mask(mask_2, "This_Photo.json");
            cv::Mat color_mask_2 = decode_segmentation_map(mask_2);

            cv::imwrite("segmented_output_2.png", color_mask_2);
            QPixmap pixmap("segmented_output_2.png");
            if (pixmap.isNull()) {
                QMessageBox::warning(this, tr(u8"Ошибка"), tr(u8"Не удалось загрузить выходное изображение."));
                return;
            }
            imageLabel->setPixmap(pixmap.scaled(imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

            cv::waitKey(0);
        }
        else {
            imageLabel->setText("Ошибка обработки U-Net.");
        }

    }
    else {
        imageLabel->setText("Ошибка обработки Segformer. Проверте исходный файл изображения");
    }
    return;
}

//Далее код для инференса
const std::map<int, cv::Scalar> CLASS_COLORS = {
    {0, cv::Scalar(0, 0, 0)},       // Фон (чёрный)
    {1, cv::Scalar(0, 0, 255)},     // Здания (красный) 
    {2, cv::Scalar(0, 255, 0)},     // Деревья (зелёный)
    {3, cv::Scalar(255, 0, 0)},     // Вода (синий)
    {4, cv::Scalar(0, 255, 255)}    // Дороги (жёлтый)
};

// Функция предобработки изображения для подачи на вход модели
cv::Mat preprocess_image(const std::string& image_path, const cv::Size& target_size) {
    // Загружаем изображение с диска
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Ошибка: не удалось загрузить изображение: " << image_path << std::endl;
        return image;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::resize(image, image, target_size);

    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    // Разбиваем изображение на отдельные каналы
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    // Нормализация по каналам (R, G, B) с использованием средних и стандартных отклонений
    channels[0] = (channels[0] - 0.485) / 0.229; // R
    channels[1] = (channels[1] - 0.456) / 0.224; // G
    channels[2] = (channels[2] - 0.406) / 0.225; // B

    // Объединяем каналы обратно в одно изображение
    cv::merge(channels, image);

    return image;
}

// Функция постобработки выходной маски
cv::Mat decode_segmentation_map(const cv::Mat& mask) {
    if (mask.empty() || mask.type() != CV_32S) {
        std::cerr << "Ошибка: неверный формат маски для декодирования." << std::endl;
        return cv::Mat();
    }

    cv::Mat color_mask(mask.size(), CV_8UC3);

    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            int class_idx = mask.at<int>(i, j);
            if (CLASS_COLORS.count(class_idx)) {
                const auto& color_scalar = CLASS_COLORS.at(class_idx);
                color_mask.at<cv::Vec3b>(i, j) = cv::Vec3b(color_scalar[0], color_scalar[1], color_scalar[2]);
            }
            else {
                color_mask.at<cv::Vec3b>(i, j) = cv::Vec3b(128, 128, 128);
            }
        }
    }
    return color_mask;
}
//Изменение размера тензора
void Interpolate(const float* input, float* output) {
    for (int y = 0; y < 512; ++y) {
        for (int x = 0; x < 512; ++x) {
            for (int c = 0; c < 5; ++c) {
                int xi = x / 4;
                int yi = y / 4;
                output[c * (512 * 512) + y * 512 + x] = input[c * (128 * 128) + yi * 128 + xi];
            }
        }
    }
}
//Совмещение двух тензоров
void Interpolate2(std::vector<float> input1, std::vector<float> input2, std::vector<float>& output) {
    for (int y = 0; y < 512; ++y) {
        for (int x = 0; x < 512; ++x) {
            for (int c = 0; c < 5; ++c) {
                output[c * (512 * 512) + y * 512 + x] = (input1[c * (512 * 512) + y * 512 + x] + input2[c * (512 * 512) + y * 512 + x]) / 2.0;
            }
        }
    }
}
// Функция выполнения инференса
std::vector<float> run_inference(Ort::Session& session, const cv::Mat& input_image) {
    const int height = input_image.rows;
    const int width = input_image.cols;
    const int channels = input_image.channels();

    std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(channels), static_cast<int64_t>(height), static_cast<int64_t>(width) };

    std::vector<float> input_tensor_values(channels * height * width);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input_tensor_values[c * (height * width) + h * width + w] = input_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = { "input" };
    const char* output_names[] = { "output" };
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

    auto& output_tensor = output_tensors.front();
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    if (output_shape.size() != 4) {
        std::cerr << "Ошибка: неожиданная форма выходного тензора." << std::endl;
        return std::vector<float>();
    }

    const int64_t num_classes = output_shape[1];
    const int64_t out_height = output_shape[2];
    const int64_t out_width = output_shape[3];
    if (out_height == 128) {
        const float* output_arr = output_tensor.GetTensorData<float>();
        std::vector<float> output_arr2(512 * 512 * 5);
        //Преобразование тензора из 128*128 в 512*512
        Interpolate(output_arr, output_arr2.data());

        return output_arr2;
    }
    else
    {
        const float* output_arr = output_tensor.GetTensorData<float>();
        std::vector<float> output_arr2(output_arr, output_arr + (512 * 512 * 5));
        return output_arr2;
    }
}
//Преобразование тензора в маску
cv::Mat inference_end(const float* output_arr) {
    cv::Mat mask(512, 512, CV_32S);
    // Для каждого пикселя выбираем класс с максимальным значением
    for (int h = 0; h < 512; ++h) {
        for (int w = 0; w < 512; ++w) {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_idx = 0;
            for (int c = 0; c < 5; ++c) {
                const float val = output_arr[c * (512 * 512) + h * 512 + w];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            mask.at<int>(h, w) = max_idx;
        }
    }
    return mask;
}