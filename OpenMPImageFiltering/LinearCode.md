```C++
#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <numbers>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>

using namespace std;

vector<vector<double>> GauseMatrix(double sigma, int radius);
void TellMyWeight(const vector<vector<double>>& inputWeightMatrix);

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        cout << "Usage: " << argv[0] << " <input> <output> <sigma> <radius>" << endl;
        return -1;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* img = nullptr;

    double sigma = atof(argv[3]);
    int radius = atoi(argv[4]);

    img = stbi_load(argv[1], &width, &height, &channels, 1);

    if (img == nullptr)
    {
        cout << "Couldn't download the image" << endl;
        return -1;
    }
    else
    {
        cout << "The image " << width << "x" << height << " was downloaded." << endl;
        cout << "Your values: " << sigma << " - sigma and " << radius << " - radius." << endl;
    }

    vector<unsigned char> result(width * height);

    vector<vector<double>> weightMatrix = GauseMatrix(sigma, radius);
    TellMyWeight(weightMatrix);

    auto start = chrono::high_resolution_clock::now();

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double newPixel = 0.0;
            double weightSum = 0.0;

            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    int yy = y + i;
                    int xx = x + j;

                    if (yy >= 0 && yy < height && xx >= 0 && xx < width)
                    {
                        int pixelIndex = yy * width + xx;

                        double weight = weightMatrix[i + radius][j + radius];

                        newPixel += img[pixelIndex] * weight;
                        weightSum += weight;
                    }
                }
            }

            int resultIndex = y * width + x;

            result[resultIndex] = static_cast<unsigned char>(newPixel / weightSum);
        }
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> time = end - start;

    cout << "Processing time: " << time.count() << " seconds" << endl;

    stbi_write_png(argv[2], width, height, 1, result.data(), width);

    cout << "Blurred image was saved as " << argv[2] << endl;

    stbi_image_free(img);
    return 0;
}

vector<vector<double>> GauseMatrix(double sigma, int radius)
{
    int dimension = 2 * radius + 1;

    vector<vector<double>> resultMatrix(dimension, vector<double>(dimension));

    double pi = numbers::pi;
    double sum = 0.0;

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            int x = i - radius;
            int y = j - radius;

            resultMatrix[i][j] =
                (1.0 / (2.0 * pi * pow(sigma, 2))) *
                exp((-(pow(x, 2) + pow(y, 2))) / (2.0 * pow(sigma, 2)));

            sum += resultMatrix[i][j];
        }
    }

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            resultMatrix[i][j] /= sum;
        }
    }

    return resultMatrix;
}

void TellMyWeight(const vector<vector<double>>& inputWeightMatrix)
{
    cout << "Your weight matrix: " << endl;

    for (int i = 0; i < inputWeightMatrix.size(); i++)
    {
        for (int j = 0; j < inputWeightMatrix.size(); j++)
        {
            cout << inputWeightMatrix[i][j] << " ";
        }

        cout << endl;
    }
}
```
