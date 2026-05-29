# MPI Gausse filter implementation

```C++
#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

vector<vector<double>> GauseMatrix(double sigma, int radius);

//input style  program.exe input.jpg output.png sigma radius
// input example  mpiexec /np 4 .\\program.exe IMG_1041.jpg blur_result.png 1.0 1
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int p;
    int r;

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    if (argc < 5)
    {
        MPI_Finalize();
        return 0;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    int ok = 1;

    double sigma = atof(argv[3]);
    int radius = atoi(argv[4]);

    unsigned char* img = nullptr;
    vector<unsigned char> result;

    if (r == 0)
    {
        img = stbi_load(argv[1], &width, &height, &channels, 1);

        if (img == nullptr)
        {
            cout << "Couldn't download the image" << endl;
            ok = 0;
        }
        else
        {
            cout << "The image " << width << "x" << height << " was downloaded." << endl;
            cout << "Sigma = " << sigma << ", radius = " << radius << endl;
        }

        if (sigma <= 0 || radius < 1)
        {
            cout << "Sigma must be > 0 and radius must be >= 1" << endl;
            ok = 0;
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!ok)
    {
        if (img != nullptr)
        {
            stbi_image_free(img);
        }

        MPI_Finalize();
        return -1;
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&radius, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<vector<double>> weightMatrix = GauseMatrix(sigma, radius);

    if (r == 0)
    {
        cout << "Your weight matrix:" << endl;

        for (int i = 0; i < weightMatrix.size(); i++)
        {
            for (int j = 0; j < weightMatrix[i].size(); j++)
            {
                cout << weightMatrix[i][j] << " ";
            }

            cout << endl;
        }
    }

    vector<int> rows(p);
    vector<int> counts(p);
    vector<int> displs(p);

    int base = height / p;
    int extra = height % p;
    int offset = 0;

    for (int i = 0; i < p; i++)
    {
        rows[i] = base;

        if (i < extra)
        {
            rows[i]++;
        }

        counts[i] = rows[i] * width;
        displs[i] = offset;

        offset += counts[i];
    }

    int localRows = rows[r];

    vector<unsigned char> localImage((localRows + 2 * radius) * width);
    vector<unsigned char> localResult(localRows * width);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatterv(
        img, counts.data(), displs.data(), MPI_UNSIGNED_CHAR,
        localImage.data() + radius * width, localRows * width, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD
    );

    if (r > 0)
    {
        MPI_Sendrecv(
            localImage.data() + radius * width,
            radius * width, MPI_UNSIGNED_CHAR,
            r - 1,
            0,
            localImage.data(),
            radius * width, MPI_UNSIGNED_CHAR,
            r - 1,
            1,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    if (r < p - 1)
    {
        MPI_Sendrecv(
            localImage.data() + localRows * width,
            radius * width,
            MPI_UNSIGNED_CHAR,
            r + 1,
            1,
            localImage.data() + (localRows + radius) * width,
            radius * width,
            MPI_UNSIGNED_CHAR,
            r + 1,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    int globalStartRow = displs[r] / width;

    for (int y = 0; y < localRows; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double newPixel = 0.0;
            double weightSum = 0.0;

            int globalY = globalStartRow + y;

            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    int yy = globalY + i;
                    int xx = x + j;

                    if (yy >= 0 && yy < height && xx >= 0 && xx < width)
                    {
                        int localY = radius + y + i;
                        int pixelIndex = localY * width + xx;

                        double weight = weightMatrix[i + radius][j + radius];

                        newPixel += localImage[pixelIndex] * weight;
                        weightSum += weight;
                    }
                }
            }

            int resultIndex = y * width + x;
            localResult[resultIndex] = static_cast<unsigned char>(newPixel / weightSum);
        }
    }

    if (r == 0)
    {
        result.resize(width * height);
    }

    MPI_Gatherv(
        localResult.data(),
        localRows * width,
        MPI_UNSIGNED_CHAR,
        result.data(),
        counts.data(),
        displs.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    double finish = MPI_Wtime();
    double localTime = finish - start;
    double maxTime = 0.0;

    MPI_Reduce(
        &localTime,
        &maxTime,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (r == 0)
    {
        stbi_write_png(argv[2], width, height, 1, result.data(), width);

        cout << "Blurred image was saved as " << argv[2] << endl;
        cout << "Processes: " << p << endl;
        cout << "Time: " << maxTime << " seconds" << endl;
    }

    if (img != nullptr)
    {
        stbi_image_free(img);
    }

    MPI_Finalize();
    return 0;
}

vector<vector<double>> GauseMatrix(double sigma, int radius)
{
    int dimension = 2 * radius + 1;

    vector<vector<double>> resultMatrix(dimension, vector<double>(dimension));

    double pi = acos(-1.0);
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
```