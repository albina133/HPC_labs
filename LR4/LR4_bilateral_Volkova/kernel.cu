#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

// без выравнивания структуры компилятором
#pragma pack(push, 1)

// файловый заголовок bmp
struct BmpFileHeader
{
    std::uint16_t bfType;
    std::uint32_t bfSize;
    std::uint16_t bfReserved1;
    std::uint16_t bfReserved2;
    std::uint32_t bfOffBits;
};

// информационный заголовок bmp
struct BmpInfoHeader
{
    std::uint32_t biSize;
    std::int32_t biWidth;
    std::int32_t biHeight;
    std::uint16_t biPlanes;
    std::uint16_t biBitCount;
    std::uint32_t biCompression;
    std::uint32_t biSizeImage;
    std::int32_t biXPelsPerMeter;
    std::int32_t biYPelsPerMeter;
    std::uint32_t biClrUsed;
    std::uint32_t biClrImportant;
};

#pragma pack(pop)

// простая структура для хранения полутонового изображения
struct GrayImage
{
    int width;
    int height;
    std::vector<unsigned char> data;

    GrayImage() : width(0), height(0) {}
};

// проверка результата cuda вызова
static void checkCuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

// ограничение значения на cpu
template <typename T>
static inline T clampHost(T value, T low, T high)
{
    return (value < low) ? low : ((value > high) ? high : value);
}

// ограничение значения на gpu. нужно при обработке границ изображения
__host__ __device__ static inline int clampDeviceInt(int value, int low, int high)
{
    return (value < low) ? low : ((value > high) ? high : value);
}

// вывод инфы о доступном gpu
static void printCudaDeviceInfo()
{
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    if (deviceCount == 0)
    {
        throw std::runtime_error("no cuda devices found");
    }

    // выбор первого доступного устройства
    int device = 0;
    checkCuda(cudaSetDevice(device), "cudaSetDevice failed");

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties failed");

    std::cout << "gpu check\n";
    std::cout << "cuda device count: " << deviceCount << "\n";
    std::cout << "selected device: " << prop.name << "\n";
    std::cout << "compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "global memory: "
        << std::fixed << std::setprecision(2)
        << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
        << " gb\n\n";
}

// чтение bmp в полутоновое изображение
static GrayImage readBmpGray(const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to open bmp file: " + filename);
    }

    BmpFileHeader fileHeader;
    BmpInfoHeader infoHeader;

    // чтение двух заголовков bmp
    file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (!file || fileHeader.bfType != 0x4D42)
    {
        throw std::runtime_error("invalid bmp file: " + filename);
    }

    if (infoHeader.biCompression != 0)
    {
        throw std::runtime_error("only uncompressed bmp is supported: " + filename);
    }

    const int width = infoHeader.biWidth;
    const int height = (infoHeader.biHeight < 0) ? -infoHeader.biHeight : infoHeader.biHeight;

    // если высота положительная, строки в bmp идут снизу вверх
    const bool bottomUp = infoHeader.biHeight > 0;
    const int bitCount = infoHeader.biBitCount;

    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error("invalid bmp size: " + filename);
    }

    // считывание палитры 8 бит
    std::vector<unsigned char> palette;
    if (bitCount == 8)
    {
        std::uint32_t colorsInPalette = infoHeader.biClrUsed ? infoHeader.biClrUsed : 256;
        palette.resize(static_cast<std::size_t>(colorsInPalette) * 4);
        file.read(reinterpret_cast<char*>(&palette[0]), static_cast<std::streamsize>(palette.size()));
        if (!file)
        {
            throw std::runtime_error("failed to read bmp palette: " + filename);
        }
    }
    // 24 и 32 бит переводятся в grayscale
    else if (bitCount != 24 && bitCount != 32)
    {
        throw std::runtime_error("only 8-bit, 24-bit and 32-bit bmp are supported: " + filename);
    }

    // переход к области пикселей
    file.seekg(fileHeader.bfOffBits, std::ios::beg);
    if (!file)
    {
        throw std::runtime_error("failed to seek to bmp pixels: " + filename);
    }

    const int bytesPerPixel = bitCount / 8;

    // каждая строка bmp выравнивается до кратности 4 байтам
    const int rowStride = ((bitCount * width + 31) / 32) * 4;
    std::vector<unsigned char> row(static_cast<std::size_t>(rowStride));

    GrayImage image;
    image.width = width;
    image.height = height;
    image.data.resize(static_cast<std::size_t>(width) * height);

    // чтение строк
    for (int rowIndex = 0; rowIndex < height; ++rowIndex)
    {
        file.read(reinterpret_cast<char*>(&row[0]), rowStride);
        if (!file)
        {
            throw std::runtime_error("failed to read bmp row: " + filename);
        }

        // приведение координаты строки к обычному порядку сверху вниз
        const int dstY = bottomUp ? (height - 1 - rowIndex) : rowIndex;

        for (int x = 0; x < width; ++x)
        {
            unsigned char gray = 0;

            if (bitCount == 8)
            {
                const unsigned char paletteIndex = row[static_cast<std::size_t>(x)];
                const std::size_t paletteOffset = static_cast<std::size_t>(paletteIndex) * 4;

                const float b = static_cast<float>(palette[paletteOffset + 0]);
                const float g = static_cast<float>(palette[paletteOffset + 1]);
                const float r = static_cast<float>(palette[paletteOffset + 2]);

                // вычисление яркости пикселя по весам каналов
                long value = std::lround(0.299f * r + 0.587f * g + 0.114f * b);
                value = clampHost<long>(value, 0L, 255L);
                gray = static_cast<unsigned char>(value);
            }
            else
            {
                const std::size_t srcIndex = static_cast<std::size_t>(x) * bytesPerPixel;
                const float b = static_cast<float>(row[srcIndex + 0]);
                const float g = static_cast<float>(row[srcIndex + 1]);
                const float r = static_cast<float>(row[srcIndex + 2]);

                // вычисление яркости пикселя по весам каналов
                long value = std::lround(0.299f * r + 0.587f * g + 0.114f * b);
                value = clampHost<long>(value, 0L, 255L);
                gray = static_cast<unsigned char>(value);
            }

            image.data[static_cast<std::size_t>(dstY) * width + x] = gray;
        }
    }

    return image;
}

// сохранение полутонового изображения в 8 бит bmp
static void writeBmpGray(const std::string& filename, int width, int height, const std::vector<unsigned char>& gray)
{
    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error("invalid grayscale image size for writing");
    }

    if (gray.size() != static_cast<std::size_t>(width) * height)
    {
        throw std::runtime_error("grayscale buffer size does not match width * height");
    }

    // ширина строки в bmp также выравнивается до 4 байт
    const int rowStride = ((width + 3) / 4) * 4;
    const std::uint32_t paletteBytes = 256 * 4;
    const std::uint32_t pixelBytes = static_cast<std::uint32_t>(rowStride * height);

    BmpFileHeader fileHeader;
    fileHeader.bfType = 0x4D42;
    fileHeader.bfOffBits = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) + paletteBytes;
    fileHeader.bfSize = fileHeader.bfOffBits + pixelBytes;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;

    BmpInfoHeader infoHeader;
    infoHeader.biSize = sizeof(BmpInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 8;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = pixelBytes;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 256;
    infoHeader.biClrImportant = 256;

    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to create bmp file: " + filename);
    }

    // запись заголовков
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    // 0 = черный, 255 = белый
    for (int i = 0; i < 256; ++i)
    {
        unsigned char entry[4];
        entry[0] = static_cast<unsigned char>(i);
        entry[1] = static_cast<unsigned char>(i);
        entry[2] = static_cast<unsigned char>(i);
        entry[3] = 0;
        file.write(reinterpret_cast<const char*>(entry), 4);
    }

    // временный буфер для одной строки
    std::vector<unsigned char> row(static_cast<std::size_t>(rowStride), 0);

    for (int rowIndex = 0; rowIndex < height; ++rowIndex)
    {
        // при записи bmp строки возврат в порядок снизу вверх
        const int srcY = height - 1 - rowIndex;

        for (int i = 0; i < rowStride; ++i)
        {
            row[static_cast<std::size_t>(i)] = 0;
        }

        // копирование одной строки пикселей в буфер строки
        for (int x = 0; x < width; ++x)
        {
            row[static_cast<std::size_t>(x)] = gray[static_cast<std::size_t>(srcY) * width + x];
        }

        file.write(reinterpret_cast<const char*>(&row[0]), rowStride);
    }
}

// cpu bilateral
static std::vector<unsigned char> applyBilateralFilterCpu(const std::vector<unsigned char>& image, int width, int height,
    float sigmaD, float sigmaR, double& processingTimeSeconds)
{
    if (image.size() != static_cast<std::size_t>(width) * height)
    {
        throw std::runtime_error("grayscale buffer size does not match width * height");
    }

    std::vector<unsigned char> output(image.size());

    const auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            unsigned char neighbors[9];
            int dxArray[9];
            int dyArray[9];
            int n = 0;

            // центральный пиксель f(a0)
            const float center = static_cast<float>(image[static_cast<std::size_t>(y) * width + x]);

            // 9 значений из окна 3 на 3
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    // на границе используется ближайший допустимый пиксель
                    const int nx = clampHost<int>(x + dx, 0, width - 1);
                    const int ny = clampHost<int>(y + dy, 0, height - 1);

                    neighbors[n] = image[static_cast<std::size_t>(ny) * width + nx];
                    dxArray[n] = dx;
                    dyArray[n] = dy;
                    ++n;
                }
            }

            float sum = 0.0f;
            float k = 0.0f;

            // вычисление нового значения пикселя по формуле фильтра
            for (int i = 0; i < 9; ++i)
            {
                const float current = static_cast<float>(neighbors[i]);

                // коэффициент, зависящий от расстояния до центрального пикселя
                const float distance2 = static_cast<float>(dxArray[i] * dxArray[i] + dyArray[i] * dyArray[i]);
                const float g = std::exp(-distance2 / (sigmaD * sigmaD));

                // коэффициент, зависящий от разницы интенсивностей
                const float diff = current - center;
                const float r = std::exp(-(diff * diff) / (sigmaR * sigmaR));

                const float weight = g * r;
                sum += current * weight;
                k += weight;
            }

            long result = std::lround(sum / k);
            result = clampHost<long>(result, 0L, 255L);
            output[static_cast<std::size_t>(y) * width + x] = static_cast<unsigned char>(result);
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    processingTimeSeconds = std::chrono::duration<double>(stop - start).count();

    return output;
}

// cuda ядро bilateral
__global__ void bilateralFilterTexture(cudaTextureObject_t texObj, unsigned char* output, int width, int height, float sigmaD, float sigmaR)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    // защита от выхода за пределы
    if (x >= width || y >= height)
    {
        return;
    }

    unsigned char neighbors[9];
    int dxArray[9];
    int dyArray[9];
    int n = 0;

    // центральный пиксель f(a0), чтение через texture memory
    const unsigned char centerUchar = tex2D<unsigned char>(texObj, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);

    const float center = static_cast<float>(centerUchar);

    // 9 значений из окна 3 на 3
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            // на границе используется ближайший допустимый пиксель
            const int nx = clampDeviceInt(x + dx, 0, width - 1);
            const int ny = clampDeviceInt(y + dy, 0, height - 1);

            // чтение соседей через texture memory
            neighbors[n] = tex2D<unsigned char>(texObj, static_cast<float>(nx) + 0.5f, static_cast<float>(ny) + 0.5f);

            dxArray[n] = dx;
            dyArray[n] = dy;
            ++n;
        }
    }

    float sum = 0.0f;
    float k = 0.0f;

    // вычисление нового значения пикселя по формуле фильтра
    for (int i = 0; i < 9; ++i)
    {
        const float current = static_cast<float>(neighbors[i]);

        // коэффициент, зависящий от расстояния до центрального пикселя
        const float distance2 = static_cast<float>(dxArray[i] * dxArray[i] + dyArray[i] * dyArray[i]);
        const float g = expf(-distance2 / (sigmaD * sigmaD));

        // коэффициент, зависящий от разницы интенсивностей
        const float diff = current - center;
        const float r = expf(-(diff * diff) / (sigmaR * sigmaR));

        const float weight = g * r;
        sum += current * weight;
        k += weight;
    }

    int result = static_cast<int>(sum / k + 0.5f);
    result = clampDeviceInt(result, 0, 255);

    output[y * width + x] = static_cast<unsigned char>(result);
}

// функция подготовки данных на gpu, запуска ядра и копирования результата обратно на cpu
static std::vector<unsigned char> applyBilateralFilterGpuTexture(const std::vector<unsigned char>& image, int width, int height,
    float sigmaD, float sigmaR, double& processingTimeSeconds)
{
    if (image.size() != static_cast<std::size_t>(width) * height)
    {
        throw std::runtime_error("grayscale buffer size does not match width * height");
    }

    cudaArray_t cuArray = 0;
    cudaTextureObject_t texObj = 0;
    unsigned char* dOutput = 0;
    cudaEvent_t startEvent = 0;
    cudaEvent_t stopEvent = 0;

    try
    {
        // описание формата одного элемента текстуры
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

        // выделение cuda массива, кот. будет использоваться как источник текстуры
        checkCuda(cudaMallocArray(&cuArray, &channelDesc, width, height), "cudaMallocArray failed");

        // копирование входного изображения в cuda массив
        checkCuda(cudaMemcpy2DToArray(
            cuArray,
            0,
            0,
            &image[0],
            static_cast<std::size_t>(width) * sizeof(unsigned char),
            static_cast<std::size_t>(width) * sizeof(unsigned char),
            height,
            cudaMemcpyHostToDevice),
            "cudaMemcpy2DToArray failed");

        // описание ресурса текстуры
        cudaResourceDesc resDesc;
        std::memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // параметры текстуры
        cudaTextureDesc texDesc;
        std::memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // создание объекта текстуры
        checkCuda(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, 0), "cudaCreateTextureObject failed");

        // выделение памяти под выходное изображение
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&dOutput), image.size() * sizeof(unsigned char)), "cudaMalloc for output failed");

        // создание cuda событий для измерения времени работы ядра
        checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start) failed");
        checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop) failed");

        // размеры блока и сетки для запуска ядра
        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        checkCuda(cudaEventRecord(startEvent, 0), "cudaEventRecord(start) failed");

        // запуск bilateral на gpu
        bilateralFilterTexture << <grid, block >> > (texObj, dOutput, width, height, sigmaD, sigmaR);

        checkCuda(cudaGetLastError(), "kernel launch failed");
        checkCuda(cudaEventRecord(stopEvent, 0), "cudaEventRecord(stop) failed");
        checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize(stop) failed");

        // вычисление времени выполнения ядра
        float elapsedMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime failed");
        processingTimeSeconds = static_cast<double>(elapsedMs) / 1000.0;

        // копирование результата обратно на cpu
        std::vector<unsigned char> output(image.size());
        checkCuda(cudaMemcpy(&output[0], dOutput, image.size() * sizeof(unsigned char), cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost failed");

        checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy(start) failed");
        startEvent = 0;
        checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy(stop) failed");
        stopEvent = 0;
        checkCuda(cudaDestroyTextureObject(texObj), "cudaDestroyTextureObject failed");
        texObj = 0;
        checkCuda(cudaFreeArray(cuArray), "cudaFreeArray failed");
        cuArray = 0;
        checkCuda(cudaFree(dOutput), "cudaFree output failed");
        dOutput = 0;

        return output;
    }
    catch (...)
    {
        if (startEvent) cudaEventDestroy(startEvent);
        if (stopEvent) cudaEventDestroy(stopEvent);
        if (texObj) cudaDestroyTextureObject(texObj);
        if (cuArray) cudaFreeArray(cuArray);
        if (dOutput) cudaFree(dOutput);
        throw;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        // сведения о gpu
        printCudaDeviceInfo();

        // имя входного изображения и значения сигма
        std::string imagePath = "thai.bmp";
        float sigmaD = 2.0f;
        float sigmaR = 60.0f;

        if (argc >= 2)
        {
            imagePath = argv[1];
        }
        if (argc >= 4)
        {
            sigmaD = static_cast<float>(std::atof(argv[2]));
            sigmaR = static_cast<float>(std::atof(argv[3]));
        }

        if (sigmaD <= 0.0f || sigmaR <= 0.0f)
        {
            throw std::runtime_error("sigma values must be positive");
        }

        // считывание исходного bmp
        GrayImage image = readBmpGray(imagePath);
        std::cout << "image loaded\n";
        std::cout << "source image size: (" << image.height << ", " << image.width << ")\n";
        std::cout << "sigma_d: " << sigmaD << "\n";
        std::cout << "sigma_r: " << sigmaR << "\n\n";

        // сохранение входного полутонового изображения для проверки
        writeBmpGray("input_gray.bmp", image.width, image.height, image.data);

        // применение bilateral на gpu
        double gpuTime = 0.0;
        std::vector<unsigned char> gpuResult = applyBilateralFilterGpuTexture(image.data, image.width, image.height, sigmaD, sigmaR, gpuTime);

        // применение bilateral на cpu
        double cpuTime = 0.0;
        std::vector<unsigned char> cpuResult = applyBilateralFilterCpu( image.data, image.width, image.height, sigmaD, sigmaR, cpuTime);

        // сохранение результатов фильтрации
        writeBmpGray("bilateral_gpu.bmp", image.width, image.height, gpuResult);
        writeBmpGray("bilateral_cpu.bmp", image.width, image.height, cpuResult);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "gpu processing time: " << gpuTime << " seconds\n";
        std::cout << "cpu processing time: " << cpuTime << " seconds\n";
        std::cout << "input image saved to: input_gray.bmp\n";
        std::cout << "gpu output image saved to: bilateral_gpu.bmp\n";
        std::cout << "cpu output image saved to: bilateral_cpu.bmp\n\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
