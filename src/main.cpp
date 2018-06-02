/**
 * Builds and trains a neural network on the MNIST dataset of handwritten digits
 * @author Aadyot Bhatnagar
 * @date April 22, 2018
 */

#include <string>
#include <cstring>
#include <iostream>
#include <string>
#include <fstream>

#include "model.hpp"
#include "MNISTParser.h"

int main(int argc, char **argv)
{
    // Read flags
    bool xfer = false;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--transfer") == 0 || strcmp(argv[i], "--t") == 0)
        {
            xfer = true;
        }
    }
    // Kind of activation to use (default relu)
    std::string activation = "relu";

    // Model Constants
    int c = 3;
    int h = 224; //400
    int w = 224;
    int n_classes = 1000;
    int in = c*h*w;
    int pos = 0;
    float *style_img = new float[in];
    float *content_img = new float[in];
    float *comb_img = new float[in];
    float *test_x = new float[in];

    std::fstream f;
    
    //Read image files into arrays
    std::cout << "Loaded training set." << std::endl;
    if (xfer){
        f.open("src/images/content.txt", std::fstream::in);
        for (std::string line; std::getline(f, line);){
            content_img[pos] = std::stof(line);
            pos++;
        }
        f.close();

        f.open("src/images/style.txt", std::fstream::in);
        for (std::string line; std::getline(f, line);){
            comb_img[pos] = std::stof(line);
            pos++;
        }
        f.close();    

        f.open("src/images/rand.txt", std::fstream::in);
        for (std::string line; std::getline(f, line);){
            comb_img[pos] = std::stof(line);
            pos++;
        }
        f.close();
    } else {
        f.open("src/images/elephant.txt", std::fstream::in);
        for (std::string line; std::getline(f, line);){
            test_x[pos] = std::stof(line);
            pos++;
        }
        f.close();

        int print_pixels = 5;
        for (int i = 0; i < print_pixels; i++){
            std::cout << test_x[i] << ", " << std::endl;
        }
        std::cout << std::endl;
        for (int i = 0; i < print_pixels; i++){
            std::cout << test_x[in-1-i] << ", " << std::endl;
        }
    }
    
    std::cout << "input NCHW: " << 1 << " " << c << " " << h << " " << w << " " << std::endl;

    bool random_weights = false; //Use weights from .h5 file
    Model *model = new Model(1, c, h, w, random_weights);

    // Model is fully defined here:
    // https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    model->add("conv", { 64, 3, 1, 1});       //(3x3xcx64)    block1_conv1
    model->add(activation);
    model->add("conv", { 64, 3, 1, 1});       //(3x3x64x64)   block1_conv2
    model->add(activation);
    model->add("mean pool", { 2 });

    model->add("conv", { 128, 3, 1, 1});       //(3x3x64x128)    block2_conv1
    model->add(activation);
    model->add("conv", { 128, 3, 1, 1});       //(3x3x128x128)   block2_conv2
    model->add(activation);
    model->add("max pool", { 2 });

    model->add("conv", { 256, 3, 1, 1});       //(3x3x128x256)   block3_conv1
    model->add(activation);
    model->add("conv", { 256, 3, 1, 1});       //(3x3x256x256)   block3_conv2
    model->add(activation);
    model->add("conv", { 256, 3, 1, 1});       //(3x3x256x256)   block3_conv3
    model->add(activation);
    model->add("max pool", { 2 });

    model->add("conv", { 512, 3, 1, 1});       //(3x3x256x512)   block4_conv1
    model->add(activation);
    model->add("conv", { 512, 3, 1, 1});       //(3x3x512x512)   block4_conv2
    model->add(activation);
    model->add("conv", { 512, 3, 1, 1});       //(3x3x512x512)   block4_conv3
    model->add(activation);
    model->add("max pool", { 2 });

    model->add("conv", { 512, 3, 1, 1});       //(3x3x512x512)   block5_conv1
    model->add(activation);
    model->add("conv", { 512, 3, 1, 1});       //(3x3x512x512)   block5_conv2
    model->add(activation);
    model->add("conv", { 512, 3, 1, 1});       //(3x3x512x512)   block5_conv3
    model->add(activation);
    model->add("max pool", { 2 });

    model->add("dense", { 4096 });
    model->add(activation);
    model->add("dense", { 4096 });
    model->add(activation);
    model->add("dense", { n_classes });
    model->add("softmax crossentropy");
    model->init_workspace();


    // Use data peeping method:
    model->check_weights();

    // USE NEURAL NETWORK
    float *test_y;
    if (xfer){
        // DO STYLE TRANSFER
        model->set_mode(0);
        test_y = model->predict(content_img, 1); //content loss
        delete[] test_y;
        
        //Save style loss metric
        model->set_mode(1);
        test_y = model->predict(style_img, 1); 
        delete[] test_y;

        //Do the Style transfer
        model->set_mode(2);
        model->transfer(comb_img, .05); // CHANGE TO TRANSFER, make func, stick in loop
    } else {
        // MAKE VGG PREDICTION
        test_y = model->predict(test_x, 1);
        // WRITE DATA TO FILE
        std::ofstream prediction_file;
        prediction_file.open("outputs/predictions.txt");

        for (int i = 0 ; i < n_classes; i++)
            prediction_file << test_y[i] << "," ;
        prediction_file << "\n";
        prediction_file.close();
        std::cout << "Wrote Predictions to file." << std::endl;
    }

    model->check_inputs();



    // Delete all dynamically allocated data

    delete model;
    delete[] test_y;

    if (xfer){
        delete[] style_img;
        delete[] content_img;
        delete[] comb_img;
    } else {
        delete[] test_x;
    }
    

    return 0;
}
