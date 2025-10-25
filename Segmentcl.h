#pragma once

#include <QMainWindow>
#include "ui_Segmentcl.h"
#include <iostream>
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QVBoxLayout>
#include <QPushButton>
#include <QWidget>
#include <QPixmap>
#include <QMessageBox>
#include <QString>
#include <string>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <map>
#include <limits>
#include <onnxruntime_cxx_api.h>


cv::Mat preprocess_image(const std::string& image_path, const cv::Size& target_size);
cv::Mat decode_segmentation_map(const cv::Mat& mask);
void Interpolate(const float* input, float* output);
void Interpolate2(std::vector<float> input1, std::vector<float> input2, std::vector<float>& output);
std::vector<float> run_inference(Ort::Session& session, const cv::Mat& input_image);
cv::Mat inference_end(const float* output_arr);



class MainWindow : public QWidget {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);

private slots:
    void openFile();
    void segFile();

private:
    QLabel* imageLabel; // Контейнер для отображения изображения
    QString SelectFile = "";
    QLabel* helloLabel;


    
};
