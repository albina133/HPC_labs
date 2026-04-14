#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

// простая структура для хранения цветного изображения
struct RGBImage
{
    int width;
    int height;
    std::vector<unsigned char> data;

    RGBImage() : width(0), height(0) {}
};

// проверка результата cuda-вызова
static void checkCuda(cudaError_t status, const char* message)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

// ограничение значения на cpu. используется при интерполяции и округлении значений пикселей
template <typename T>
static inline T clampHost(T value, T low, T high)
{
    return (value < low) ? low : ((value > high) ? high : value);
}

// функция ограничения. нужна при обработке границ изображения внутри ядра
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

// чтение bmp и преобразование в rgb
static RGBImage readBmp(const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to open bmp file: " + filename);
    }

    BmpFileHeader fileHeader;
    BmpInfoHeader infoHeader;

    // чтение двух bmp
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
    // 24 и 32 бит
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

    RGBImage image;
    image.width = width;
    image.height = height;

    // изображение как последовательность rgb
    image.data.resize(static_cast<std::size_t>(width) * height * 3);

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
            const std::size_t dstIndex = (static_cast<std::size_t>(dstY) * width + x) * 3;

            if (bitCount == 8)
            {
                const unsigned char paletteIndex = row[static_cast<std::size_t>(x)];
                const std::size_t paletteOffset = static_cast<std::size_t>(paletteIndex) * 4;
                image.data[dstIndex + 0] = palette[paletteOffset + 2];
                image.data[dstIndex + 1] = palette[paletteOffset + 1];
                image.data[dstIndex + 2] = palette[paletteOffset + 0];
            }
            else
            {
                const std::size_t srcIndex = static_cast<std::size_t>(x) * bytesPerPixel;
                const unsigned char b = row[srcIndex + 0];
                const unsigned char g = row[srcIndex + 1];
                const unsigned char r = row[srcIndex + 2];
                image.data[dstIndex + 0] = r;
                image.data[dstIndex + 1] = g;
                image.data[dstIndex + 2] = b;
            }
        }
    }

    return image;
}

// сохранение цветного изображ в 24 бит bmp
static void writeBmpRgb(const std::string& filename, int width, int height, const std::vector<unsigned char>& rgb)
{
    if (width <= 0 || height <= 0)
    {
        throw std::runtime_error("invalid rgb image size for writing");
    }

    if (rgb.size() != static_cast<std::size_t>(width) * height * 3)
    {
        throw std::runtime_error("rgb buffer size does not match width * height * 3");
    }

    // ширина строки в bmp также выравнивается до 4 байт
    const int rowStride = ((width * 3 + 3) / 4) * 4;
    const std::uint32_t pixelBytes = static_cast<std::uint32_t>(rowStride * height);

    BmpFileHeader fileHeader;
    fileHeader.bfType = 0x4D42;
    fileHeader.bfOffBits = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader);
    fileHeader.bfSize = fileHeader.bfOffBits + pixelBytes;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;

    BmpInfoHeader infoHeader;
    infoHeader.biSize = sizeof(BmpInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = pixelBytes;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to create bmp file: " + filename);
    }

    // запись заголовков
    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

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

        for (int x = 0; x < width; ++x)
        {
            const std::size_t srcIndex = (static_cast<std::size_t>(srcY) * width + x) * 3;
            const std::size_t dstIndex = static_cast<std::size_t>(x) * 3;

            // преобразование rgb обратно в bgr для bmp
            row[dstIndex + 0] = rgb[srcIndex + 2];
            row[dstIndex + 1] = rgb[srcIndex + 1];
            row[dstIndex + 2] = rgb[srcIndex + 0];
        }

        file.write(reinterpret_cast<const char*>(&row[0]), rowStride);
    }
}

// сохранение полутонового изображения в 8 бит bmp в сером
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
    if (!file) {
        throw std::runtime_error("failed to create bmp file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    // 0 = черный, 255 = белый, база
    for (int i = 0; i < 256; ++i)
    {
        unsigned char entry[4];
        entry[0] = static_cast<unsigned char>(i);
        entry[1] = static_cast<unsigned char>(i);
        entry[2] = static_cast<unsigned char>(i);
        entry[3] = 0;
        file.write(reinterpret_cast<const char*>(entry), 4);
    }

    std::vector<unsigned char> row(static_cast<std::size_t>(rowStride), 0);

    for (int rowIndex = 0; rowIndex < height; ++rowIndex)
    {
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

// функция изменения размера фото. тестировалась, но не используется
static RGBImage resizeBilinearRgb(const RGBImage& src, int dstWidth, int dstHeight)
{
    if (src.width <= 0 || src.height <= 0 || src.data.empty())
    {
        throw std::runtime_error("empty source image for resize");
    }

    RGBImage dst;
    dst.width = dstWidth;
    dst.height = dstHeight;
    dst.data.resize(static_cast<std::size_t>(dstWidth) * dstHeight * 3);

    const float scaleX = static_cast<float>(src.width) / static_cast<float>(dstWidth);
    const float scaleY = static_cast<float>(src.height) / static_cast<float>(dstHeight);

    for (int y = 0; y < dstHeight; ++y)
    {
        float fy = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        float wy = fy - static_cast<float>(y0);

        y0 = clampHost<int>(y0, 0, src.height - 1);
        y1 = clampHost<int>(y1, 0, src.height - 1);

        for (int x = 0; x < dstWidth; ++x) {
            float fx = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            float wx = fx - static_cast<float>(x0);

            x0 = clampHost<int>(x0, 0, src.width - 1);
            x1 = clampHost<int>(x1, 0, src.width - 1);

            // билинейная интерполяция отдельно для каждого канала
            for (int c = 0; c < 3; ++c)
            {
                const float p00 = static_cast<float>(src.data[(static_cast<std::size_t>(y0) * src.width + x0) * 3 + c]);
                const float p10 = static_cast<float>(src.data[(static_cast<std::size_t>(y0) * src.width + x1) * 3 + c]);
                const float p01 = static_cast<float>(src.data[(static_cast<std::size_t>(y1) * src.width + x0) * 3 + c]);
                const float p11 = static_cast<float>(src.data[(static_cast<std::size_t>(y1) * src.width + x1) * 3 + c]);

                const float top = p00 + (p10 - p00) * wx;
                const float bottom = p01 + (p11 - p01) * wx;
                const float value = top + (bottom - top) * wy;

                long rounded = std::lround(value);
                rounded = clampHost<long>(rounded, 0L, 255L);

                dst.data[(static_cast<std::size_t>(y) * dstWidth + x) * 3 + c] =
                    static_cast<unsigned char>(rounded);
            }
        }
    }

    return dst;
}

// перевод rgb в grayscale
static std::vector<unsigned char> rgbToGrayLikeNotebook(const RGBImage& image)
{
    std::vector<unsigned char> gray(static_cast<std::size_t>(image.width) * image.height);

    for (int y = 0; y < image.height; ++y)
    {
        for (int x = 0; x < image.width; ++x)
        {
            const std::size_t idx = (static_cast<std::size_t>(y) * image.width + x) * 3;
            const float r = static_cast<float>(image.data[idx + 0]);
            const float g = static_cast<float>(image.data[idx + 1]);
            const float b = static_cast<float>(image.data[idx + 2]);

            // вычисление яркости пикселя по весам каналов
            const float grayValue = 0.114f * r + 0.587f * g + 0.299f * b;
            long rounded = std::lround(grayValue);
            rounded = clampHost<long>(rounded, 0L, 255L);

            gray[static_cast<std::size_t>(y) * image.width + x] = static_cast<unsigned char>(rounded);
        }
    }

    return gray;
}

// функция добавление шума
static std::vector<unsigned char> addSaltAndPepperNoise(const std::vector<unsigned char>& image, int width, int height, double probability)
{
    std::vector<unsigned char> noisyImage = image;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> probDist(0.0, 1.0);
    std::uniform_int_distribution<int> valueDist(0, 255);
    std::uniform_int_distribution<int> blackOrWhiteDist(0, 1);

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            // с заданной вероятностью замена тек. пикселя на шум
            if (probDist(rng) <= probability)
            {
                int nois = valueDist(rng);
                (void)nois;

                // блек ор вайт
                int blackOrWhite = blackOrWhiteDist(rng);
                noisyImage[static_cast<std::size_t>(i) * width + j] = blackOrWhite ? 255 : 0;
            }
        }
    }

    return noisyImage;
}

// cuda-ядро медианного фильтра 3x3
__global__ void medianFilterTexture(cudaTextureObject_t texObj, unsigned char* output, int width, int height)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    // защита от выхода за пределы
    if (x >= width || y >= height)
    {
        return;
    }

    // 9 значений из окна 3 на 3
    unsigned char neighbors[9];
    int n = 0;

    // чтение соседей через texture memory
    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            // на границе используется ближайший допустимый пиксель
            const int nx = clampDeviceInt(x + i, 0, width - 1);
            const int ny = clampDeviceInt(y + j, 0, height - 1);

            // tex2d читает значение из текстуры по координатам
            neighbors[n] = tex2D<unsigned char>(texObj, static_cast<float>(nx) + 0.5f, static_cast<float>(ny) + 0.5f);
            ++n;
        }
    }

    // сортировка 9 элементов пузырьком. после сортировки медиана будет находиться в центре массива
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9 - i - 1; ++j)
        {
            if (neighbors[j] > neighbors[j + 1])
            {
                unsigned char temp = neighbors[j];
                neighbors[j] = neighbors[j + 1];
                neighbors[j + 1] = temp;
            }
        }
    }

    // элемент с индексом 4 является медианой для девяти чисел
    output[y * width + x] = neighbors[4];
}

// функция подготовки данных на gpu, запуска ядра и копирования результата обратно на cpu
static std::vector<unsigned char> applyMedianFilterGpuTexture(const std::vector<unsigned char>& image, int width, int height, double& processingTimeSeconds)
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

    try {
        // описание формата одного элемента текстуры
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

        // выделение cuda массива, кот. будет использоваться как источник текстуры
        checkCuda(cudaMallocArray(&cuArray, &channelDesc, width, height), "cudaMallocArray failed");

        // копирование входного изображение в cuda массив
        checkCuda(cudaMemcpy2DToArray(cuArray,
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

        // создание cuda события для измерения времени работы ядра
        checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start) failed");
        checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop) failed");

        // размеры блока и сетки для запуска ядра
        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        checkCuda(cudaEventRecord(startEvent, 0), "cudaEventRecord(start) failed");

        // запуск фильтра на gpu
        medianFilterTexture << <grid, block >> > (texObj, dOutput, width, height);

        checkCuda(cudaGetLastError(), "kernel launch failed");
        checkCuda(cudaEventRecord(stopEvent, 0), "cudaEventRecord(stop) failed");
        checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize(stop) failed");

        // вычисление времени выполнения ядра
        float elapsedMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime failed");
        processingTimeSeconds = static_cast<double>(elapsedMs) / 1000.0;

        // копирование реза обратно на cpu
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

int main()
{
    try
    {
        // сведения о gpu
        printCudaDeviceInfo();

        // ИМЯ входного изображения
        const std::string imagePath = "thai.bmp";

        // считывание исходного bmp
        RGBImage sourceImage = readBmp(imagePath);
        std::cout << "image loaded\n";
        std::cout << "source image size: (" << sourceImage.height << ", " << sourceImage.width << ", 3)\n";

        // rgb копия
        writeBmpRgb("step_rgb_original.bmp", sourceImage.width, sourceImage.height, sourceImage.data);

        // перевод изображения в оттенки серого
        std::vector<unsigned char> grayImage = rgbToGrayLikeNotebook(sourceImage);
        writeBmpGray("gray_image.bmp", sourceImage.width, sourceImage.height, grayImage);

        // добавляем шума (вероятность 0.1 пусть)
        std::vector<unsigned char> noisyImage = addSaltAndPepperNoise(grayImage, sourceImage.width, sourceImage.height, 0.1);
        writeBmpGray("input_image.bmp", sourceImage.width, sourceImage.height, noisyImage);

        // применение медианного фильтра на gpu
        double processingTime = 0.0;
        std::vector<unsigned char> outputImage = applyMedianFilterGpuTexture(noisyImage,
            sourceImage.width,
            sourceImage.height,
            processingTime);

        // сохранение реза фильтрации
        writeBmpGray("output_image.bmp", sourceImage.width, sourceImage.height, outputImage);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "processing time: " << processingTime << " seconds\n";
        std::cout << "output image saved to: output_image.bmp\n\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
