// Главный файл симулятора аппаратного обеспечения СЗМ
// 
// Этот файл является точкой входа для симулятора сканирующего 
// зондового микроскопа на C++.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>

// Класс для представления поверхности
class SurfaceModel {
private:
    std::vector<std::vector<double>> heightMap;
    int width, height;
    
public:
    // Конструктор модели поверхности
    SurfaceModel(int w, int h) : width(w), height(h) {
        heightMap.resize(height, std::vector<double>(width, 0.0));
        generateSurface();
    }
    
    // Генерирует случайную поверхность с заданными характеристиками
    void generateSurface() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 0.5);
        
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                heightMap[i][j] = dis(gen);
            }
        }
        
        // Добавляем несколько "кратеров" и "гор"
        addFeatures();
    }
    
    // Добавляет искусственные особенности к поверхности
    void addFeatures() {
        // Добавляем несколько круглых кратеров
        for (int crater = 0; crater < 3; ++crater) {
            int centerX = rand() % width;
            int centerY = rand() % height;
            double radius = 5.0 + rand() % 10;
            
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    double dist = sqrt(pow(i - centerY, 2) + pow(j - centerX, 2));
                    if (dist <= radius) {
                        double depth = -0.5 * (1 - dist/radius);
                        heightMap[i][j] += depth;
                    }
                }
            }
        }
        
        // Добавляем несколько "гор"
        for (int mountain = 0; mountain < 2; ++mountain) {
            int centerX = rand() % width;
            int centerY = rand() % height;
            double radius = 4.0 + rand() % 8;
            
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    double dist = sqrt(pow(i - centerY, 2) + pow(j - centerX, 2));
                    if (dist <= radius) {
                        double height = 0.7 * (1 - dist/radius);
                        heightMap[i][j] += height;
                    }
                }
            }
        }
    }
    
    // Получает высоту в заданной точке
    double getHeight(int x, int y) const {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            return heightMap[y][x];
        }
        return 0.0; // Значение по умолчанию за пределами поверхности
    }
    
    // Сохраняет модель поверхности в файл
    void saveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << width << " " << height << "\n";
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    file << heightMap[i][j] << " ";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Модель поверхности сохранена в файл: " << filename << std::endl;
        } else {
            std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
        }
    }
    
    // Возвращает ширину поверхности
    int getWidth() const { return width; }
    
    // Возвращает высоту поверхности
    int getHeight() const { return height; }
};

// Класс для модели зонда
class ProbeModel {
private:
    double x, y, z; // Позиция зонда
    double scanSpeed; // Скорость сканирования
    double maxZ; // Максимальная высота зонда
    
public:
    // Конструктор модели зонда
    ProbeModel() : x(0), y(0), z(10.0), scanSpeed(0.1), maxZ(20.0) {}
    
    // Устанавливает позицию зонда
    void setPosition(double newX, double newY, double newZ) {
        x = newX;
        y = newY;
        z = newZ;
    }
    
    // Получает текущую позицию зонда
    std::vector<double> getPosition() const {
        return {x, y, z};
    }
    
    // Перемещает зонд к следующей позиции
    void moveTo(double targetX, double targetY, double targetZ) {
        x = targetX;
        y = targetY;
        z = targetZ;
    }
    
    // Адаптирует высоту зонда к поверхности
    double adjustToSurface(const SurfaceModel& surface) {
        double surfaceHeight = surface.getHeight(static_cast<int>(x), static_cast<int>(y));
        double adjustedZ = surfaceHeight + 0.5; // Поддерживаем небольшой зазор
        
        if (adjustedZ > maxZ) {
            adjustedZ = maxZ;
        } else if (adjustedZ < 0.1) {
            adjustedZ = 0.1; // Минимальная безопасная высота
        }
        
        z = adjustedZ;
        return z;
    }
};

// Класс для контроллера СЗМ
class SPMController {
private:
    SurfaceModel* surface;
    ProbeModel probe;
    std::vector<std::vector<double>> scanData; // Данные сканирования
    int currentX, currentY; // Текущая позиция сканирования
    
public:
    // Конструктор контроллера СЗМ
    SPMController() : surface(nullptr), currentX(0), currentY(0) {}
    
    // Устанавливает модель поверхности для сканирования
    void setSurface(SurfaceModel* surf) {
        surface = surf;
        scanData.clear();
        scanData.resize(surface->getHeight(), std::vector<double>(surface->getWidth()));
    }
    
    // Выполняет сканирование всей поверхности
    void scanSurface() {
        if (!surface) {
            std::cerr << "Ошибка: Модель поверхности не установлена!" << std::endl;
            return;
        }
        
        std::cout << "Начинаем сканирование поверхности..." << std::endl;
        
        int width = surface->getWidth();
        int height = surface->getHeight();
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Устанавливаем позицию зонда
                probe.moveTo(x, y, 0); // Z будет адаптирован автоматически
                
                // Адаптируем высоту зонда к поверхности
                double adjustedZ = probe.adjustToSurface(*surface);
                
                // Сохраняем данные сканирования (высота зонда как индикатор рельефа)
                scanData[y][x] = adjustedZ;
                
                currentX = x;
                currentY = y;
                
                // Выводим прогресс каждые 10 точек
                if ((x + y * width) % (width * height / 10) == 0) {
                    std::cout << "Прогресс: " << (x + y * width) * 100 / (width * height) << "%" << std::endl;
                }
            }
        }
        
        std::cout << "Сканирование завершено!" << std::endl;
    }
    
    // Сохраняет результаты сканирования в файл
    void saveScanResults(const std::string& filename) const {
        if (scanData.empty()) {
            std::cerr << "Ошибка: Нет данных сканирования для сохранения!" << std::endl;
            return;
        }
        
        std::ofstream file(filename);
        if (file.is_open()) {
            int height = scanData.size();
            int width = scanData[0].size();
            
            file << width << " " << height << "\n";
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    file << scanData[i][j] << " ";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Результаты сканирования сохранены в файл: " << filename << std::endl;
        } else {
            std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
        }
    }
    
    // Получает ширину модели поверхности
    int getWidth() const {
        return surface ? surface->getWidth() : 0;
    }
    
    // Получает высоту модели поверхности
    int getHeight() const {
        return surface ? surface->getHeight() : 0;
    }
};

// Главная функция программы
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    СИМУЛЯТОР АППАРАТНОГО ОБЕСПЕЧЕНИЯ СЗМ" << std::endl;
    std::cout << "         (Scanning Probe Microscope)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Создаем модель поверхности 20x20
    SurfaceModel surface(20, 20);
    std::cout << "Создана модель поверхности размером " << surface.getWidth() 
              << "x" << surface.getHeight() << std::endl;
    
    // Сохраняем модель поверхности
    surface.saveToFile("surface_model.txt");
    
    // Создаем контроллер СЗМ и устанавливаем поверхность
    SPMController controller;
    controller.setSurface(&surface);
    
    // Выполняем сканирование
    controller.scanSurface();
    
    // Сохраняем результаты сканирования
    controller.saveScanResults("scan_results.txt");
    
    std::cout << "Симуляция завершена. Результаты сохранены." << std::endl;
    
    return 0;
}