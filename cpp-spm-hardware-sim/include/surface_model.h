#ifndef SURFACE_MODEL_H
#define SURFACE_MODEL_H

#include <vector>
#include <string>

/**
 * @brief Класс для представления модели поверхности
 * 
 * Обрабатывает генерацию и загрузку данных о топографии поверхности. 
 * Поддерживает как процедурную генерацию, так и загрузку на основе файлов.
 */
class SurfaceModel {
private:
    std::vector<std::vector<double>> heightMap; ///< Карта высот поверхности
    int width;  ///< Ширина поверхности
    int height; ///< Высота поверхности

public:
    /**
     * @brief Конструктор модели поверхности
     * @param w Ширина поверхности
     * @param h Высота поверхности
     */
    SurfaceModel(int w, int h);

    /**
     * @brief Деструктор модели поверхности
     */
    ~SurfaceModel();

    /**
     * @brief Генерирует случайную поверхность с заданными характеристиками
     */
    void generateSurface();

    /**
     * @brief Добавляет искусственные особенности к поверхности
     */
    void addFeatures();

    /**
     * @brief Получает высоту в заданной точке
     * @param x Координата X
     * @param y Координата Y
     * @return Высота в точке (x,y)
     */
    double getHeight(int x, int y) const;

    /**
     * @brief Сохраняет модель поверхности в файл
     * @param filename Имя файла для сохранения
     */
    void saveToFile(const std::string& filename) const;

    /**
     * @brief Возвращает ширину поверхности
     * @return Ширина поверхности
     */
    int getWidth() const { return width; }

    /**
     * @brief Возвращает высоту поверхности
     * @return Высота поверхности
     */
    int getHeight() const { return height; }
};

#endif // SURFACE_MODEL_H